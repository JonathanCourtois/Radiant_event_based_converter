# -----------------------------------------------------------------------------
# Author:   Jonathan COURTOIS
# Email:    contact@jonathancourtois.com
# Date:     march 25, 2025
# Description: Give an event based camera converter for video RGB images.
# -----------------------------------------------------------------------------

import torch
import numpy as np

class cam_evb():
    def __init__(self, threshold = 0., input_resolution = (1,128,128), noise_level = None, multi_threshold:bool = False) -> None:
        """
        Initialize the camera event-based class.
        Args:
            threshold           (float): The minimum amount of 'light intensity variation' after log scale, required to trigger a spike.
            input_resolution    (tuple): The resolution of the input data (c,h,w).(default : (1,128,128))
            noise_level         (float): The level of noise in the input data. If None, no noise is added.(default=None)
            multi_threshold     (bool) : Alow the spike matrix to store more than one spike per pixel. Result in non binary spike matrix.(default=False)
        """
        self.resolution = input_resolution
        self.threshold  = threshold
        if torch.cuda.is_available():
            print("Using GPU")
            device = 'cuda'
        else:
            print("Using CPU")
            device = 'cpu'
        
        self.grid_map   = torch.zeros((self.resolution[0],self.resolution[1],self.resolution[2]),requires_grad=False, device=device)
        self.spike      = torch.zeros((int(2*self.resolution[0]),self.resolution[1],self.resolution[2]),requires_grad=False, device=device)
        self.noise      = noise_level    
        self.device     = device

        self.multi_threshold    = multi_threshold
        self.spike_mat_max      = 0

    def out(self):
        """
        Return the spike features map.
        """
        return self.spike

    def out_rgb(self, color="blue", spike_frame=None):
        """
        Return the spike features map in RGB format.
        Args:
           color (str): The color of the spike features map. Can be "white" or "blue".
           spike_frame (np.ndarray): The spike features map in spike format. If not provided, the spike features map is converted from the camera grid_map.
        """
        return self.convert_event_frame_to_RGB(color=color, spike_frame=spike_frame)
    
    def convert_event_frame_to_RGB(self, color="blue", spike_frame=None):
        """
        Convert the event frame to RGB format.
        Args:
           color (str): The color of the spike features map. Can be "white" or "blue" or "grad"(for multi threshold).
           spike_frame (np.ndarray): The spike features map in spike format. If not provided, the spike features map is converted from the camera grid_map.
        Returns:
           np.ndarray: The spike features map with shape.(height, width,3)
        """
        if spike_frame is not None:
            spike = spike_frame
        else:
            spike = self.spike

        if color == "white":
            background = 255
            c_pos = (0, 0, 255)
            c_neg = [255, 0, 0]
        elif color == "grad":
            # color == "blue":
            background = 0
            c_pos = (255,255,255)
            c_neg = [225,105, 64]
        else :
            # color == "blue":
            background = 0
            c_pos = (255,255,255)
            c_neg = [225,105, 64]
            
        x = spike.shape[-2]
        y = spike.shape[-1]
        # Create a white image
        white = torch.full((x, y, 3), background, requires_grad=False)
        if color == "grad":
            # Pushing pos event to 128-255 and neg event to 0-127
            pos_rgb     = spike[[0,2,4],:,:]
            neg_rgb     = spike[[1,3,5],:,:]
            merge_rgb   = pos_rgb-neg_rgb
            merge_rgb[merge_rgb>0] = merge_rgb[merge_rgb>0]/self.spike_mat_max*127 + 128
            merge_rgb[merge_rgb<0] = merge_rgb[merge_rgb<0]/self.spike_mat_max*127
            # white = merge_rgb.astype(torch.uint8).transpose(1,2,0)
            white = merge_rgb.permute(1,2,0)
        else:
            # Add positive color dot to the image
            # whit shape x,y,3 where x,y
            # Add negative color dot to the image
            white[spike[1]>=1] = c_neg

        return white

    def update(self, img):
        """
        Update the camera grid map based on the input image.
        input img must be dim (c,height,width)
        """
        # add gaussian noise on img
        if self.noise is not None:
            # tmp = np.random.normal(0, self.noise, img.shape)
            tmp = torch.randn_like(img)*self.noise
            tmp.to(self.device)
            offset = 3e-10
            tmp[tmp<offset] = 0
            
            img = img + tmp
        img[img<=0] = 1e-10
                    
        pos = ((torch.log(img) - self.grid_map) >  self.threshold)
        neg = ((torch.log(img) - self.grid_map) < -self.threshold)
        for i in range(self.resolution[0]):
            # print(f"{self.spike.shape} {pos.shape} {neg.shape}")
            if self.multi_threshold:
                self.spike[i*2]     = pos[i]*((torch.log(img[i]) - self.grid_map[i])*((torch.log(img[i]) - self.grid_map[i]) >  self.threshold)//self.threshold)
                self.spike[i*2+1]   = neg[i]*((torch.log(img[i]) - self.grid_map[i])*((torch.log(img[i]) - self.grid_map[i]) < -self.threshold)//-self.threshold)
            else:
                self.spike[i*2]     = pos[i]*1
                self.spike[i*2+1]   = neg[i]*1
        self.grid_map[pos] = torch.log(img)[pos]
        self.grid_map[neg] = torch.log(img)[neg]
        
        if self.spike.max() > self.spike_mat_max:
            self.spike_mat_max = self.spike.max()
        return

    def rgb_to_hsv(self, rgb_img):
        """
        Convert RGB image to HSV format.
        Input: rgb_img - numpy array of shape (height, width, 3) with values in range [0, 255]
        Output: HSV image as numpy array of shape (height, width, 3)
                H in range [0, 360], S and V in range [0, 1]
        """
        # Normalize RGB values to range [0, 1]
        rgb_normalized = rgb_img.type(torch.float32) / 255.0
        
        # Reshape the image to 2D array of pixels
        pixels = rgb_normalized.reshape(-1, 3)
        
        # Extract R, G, B values
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        
        # Calculate Value (V)
        v = torch.max(pixels, axis=1)
        
        # Calculate Saturation (S)
        delta = v - torch.min(pixels, axis=1)
        s = torch.where(v != 0, delta/v, 0)
        
        # Calculate Hue (H)
        h = torch.zeros_like(v)
        
        # When v == delta, it means only one color has non-zero value
        non_zero_delta_mask = (delta != 0)
        
        # Red is maximum
        red_max_mask = (v == r) & non_zero_delta_mask
        h[red_max_mask] = 60 * ((g[red_max_mask] - b[red_max_mask]) / delta[red_max_mask])
        
        # Green is maximum
        green_max_mask = (v == g) & non_zero_delta_mask
        h[green_max_mask] = 60 * (2 + (b[green_max_mask] - r[green_max_mask]) / delta[green_max_mask])
        
        # Blue is maximum
        blue_max_mask = (v == b) & non_zero_delta_mask
        h[blue_max_mask] = 60 * (4 + (r[blue_max_mask] - g[blue_max_mask]) / delta[blue_max_mask])
        
        # Make sure hue is in [0, 360]
        h = h % 360
        
        # Stack the HSV channels and reshape back to original image shape
        hsv = torch.stack([h, s, v], dim=1)
        hsv_img = hsv.reshape(rgb_img.shape)
        
        return hsv_img