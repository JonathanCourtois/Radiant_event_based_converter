# -----------------------------------------------------------------------------
# Author:   Jonathan COURTOIS
# Email:    contact@jonathancourtois.com
# Date:     march 25, 2025
# Description: Give an event based camera converter for video RGB images.
# -----------------------------------------------------------------------------

import tqdm
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from src.camera_eventbased import cam_evb
import argparse

def get_cv2_infos(cap):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = int(cap.get(cv2.CAP_PROP_FPS))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, fps, width, height

def merge_channel_to_polarity(spike_video_frames):
        # we can then choose to have a 2*x*y images shape merging pos/neg polarity
        pos_rgb = spike_video_frames[:,[0,2,4],:,:].sum(axis=1)
        neg_rgb = spike_video_frames[:,[1,3,5],:,:].sum(axis=1)
        spike_video_frames_rgb_merged = np.stack([pos_rgb,neg_rgb],axis=1)
        return spike_video_frames_rgb_merged

def merge_polarity_to_rgb(spike_video_frames):
        # we can then choose to have a 3*x*y images shape with polarity merged in -min and +max
        red     = spike_video_frames[:,0,:,:]-spike_video_frames[:,1,:,:]
        green   = spike_video_frames[:,2,:,:]-spike_video_frames[:,3,:,:]
        blue    = spike_video_frames[:,4,:,:]-spike_video_frames[:,5,:,:]
        spike_video_frames_rgb_merged = np.stack([red,green,blue],axis=1)
        return spike_video_frames_rgb_merged

def convert_mp4_video(path_to_video, output_path=None, event_camera=None, merge_method="first_channel", args=None):
    """
    Convert RGB mp4 Video to Event Based like mp4 video."

    merge_method : str :    "first_channel"
                            "polarity" (channel sumed to polarity)
                            "channel"  (rgb channels sumed to polarity)
    """

    # Open the video file
    cap = cv2.VideoCapture(path_to_video)
    frame_count, fps, width, height = get_cv2_infos(cap)
    # Print the video information
    print(f"Video informations: {frame_count} frames, {fps} fps, {width}x{height} pixels\nPath: {path_to_video}")

    if event_camera is None:
        event_camera = cam_evb(threshold=0.1, input_resolution=(3,height,width), noise_level=0.1)
        print("No event camera provided, using default event camera.")

    ret = True
    i   = 0
    
    spike_video_frames  = []
    rgb_video_frames    = []

    with tqdm.tqdm(total=frame_count) as pbar:
        while ret:
            ret, image  = cap.read() # image is a numpy array with shape (height, width, 3)
            if args.input_rate is not None:
                if args.input_rate < 1:
                    raise ValueError("input_rate must be greater than 1.")
                if i%args.input_rate != 0: # Skip some frames to match frequency in ideal video without frequency issue there is no need to do this
                    i += 1
                    pbar.update(1)
                    continue
            if not ret:
                break

            event_camera.update(image.transpose(2,0,1))
            spike_mat = event_camera.spike
            spike_video_frames.append(spike_mat)

            if merge_method == "first_channel":
                # using out_rgb tool that use the first color channel only (R) to pos/neg polarity in v1.0
                rgb_out = event_camera.out_rgb()
            elif merge_method == "polarity":
                # display result with channel sumed to polarity "c_to_p"
                c_to_p = merge_channel_to_polarity(np.array([spike_mat]))[0]
                rgb_out = event_camera.convert_event_frame_to_RGB(color="blue", spike_frame=c_to_p)
            elif merge_method == "channel":
                # display result with polarity merged to rgb channels "p_to_c"
                rgb_out = event_camera.convert_event_frame_to_RGB(color="grad", spike_frame=spike_mat)
            else:
                raise ValueError(f"merge_method {merge_method} not supported.") 
                
            rgb_video_frames.append(rgb_out)
            i += 1
            pbar.update(1)
        cap.release()
        cv2.destroyAllWindows()
        pbar.close()
    

    # Save the video
    print(f"fps: {fps} frame_count: {frame_count} rgb_video_frames: {len(rgb_video_frames)}")
    if args is not None and args.fps is not None:
        fps = args.fps
        print(f"fps adjusted to {fps}")
    elif frame_count != len(rgb_video_frames):
         # adjust the fps to match the number of frames
        fps = frame_count/len(rgb_video_frames)
        print(f"fps adjusted to {fps}")

    if output_path is not None:
        print(f"Saving video to {output_path}")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (rgb_video_frames[0].shape[1], rgb_video_frames[0].shape[0]))
        for frame in rgb_video_frames:
            out.write(frame)
        out.release()

    if args is not None and args.save_spike_mat:
        spike_video_frames = np.array(spike_video_frames)
        if output_path is None:
            output_path = "./.mp4"
        spike_mat_path = output_path.replace(".mp4", "_spike_mat.npy")
        print(f"Saving spike matrix to {spike_mat_path}")
        np.save(spike_mat_path, spike_video_frames)

parser = argparse.ArgumentParser(description='Convert RGB video to Event Based video.')
parser.add_argument('video_path',       type=str,   help='Path to the video to convert.')
parser.add_argument('--output_path',    type=str,   default="./video_out/",   help='Path to the output video.')
parser.add_argument('--threshold',      type=float, default=0.3,    help='Threshold for the event camera.')
parser.add_argument('--noise_level',    type=float, default=0.8,    help='Noise level for the event camera.')
parser.add_argument('--merge_method',   type=str,   default="first_channel", help='Method to merge the channels to RGB. Can Be "first_channel", "polarity" or "channel"')
parser.add_argument('--fps',            type=int,   default=None,   help='Fps of the output video.')
parser.add_argument('--input_rate',     type=int, default=None,        help='Input rate of the video (convert every n frame).')
parser.add_argument('--save_spike_mat', action='store_true', help='Save the spike matrix as npy file.')
args = parser.parse_args()


def main():
    video_path_test = args.video_path

    video_name = video_path_test.split("/")[-1].split(".")[0]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    out_path   = os.path.join(args.output_path, f"EVB_{video_name}_{args.merge_method}.mp4")

    cap = cv2.VideoCapture(video_path_test)
    frame_count, fps, width, height = get_cv2_infos(cap)
    video_res = (3, height, width)
    cap.release()

    event_camera = cam_evb(threshold=args.threshold, input_resolution=video_res, noise_level=args.noise_level, multi_threshold=True)

    convert_mp4_video(video_path_test, output_path=out_path, event_camera=event_camera, merge_method=args.merge_method, args=args)

if __name__ == "__main__":
    main()