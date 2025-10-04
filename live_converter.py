# -----------------------------------------------------------------------------
# Author:   Jonathan COURTOIS
# Email:    contact@jonathancourtois.com
# Date:     october 25, 2025
# Description: live converter of usb camera to event based like video.
# -----------------------------------------------------------------------------

import argparse
import os
import time
import cv2
import torch
import numpy as np
from src.camera_eventbased import cam_evb


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def run_camera(camera_index: int = 0,
               output_dir: str = "./video_out",
               initial_threshold: float = 0.3,
               initial_noise: float = 0.8,
               merge_methods=None,
               multi_threshold: bool = False,
               target_fps: int | None = None):
    """
    Open a USB camera, feed frames to cam_evb, show the result and accept keyboard commands:
      q : quit
      SPACE : toggle recording
      m : cycle merge method (visualization color)
      y : increase threshold
      r : decrease threshold
      n : increase noise
      b : decrease noise
    """
    if merge_methods is None:
        merge_methods = ["blue", "white", "grad"]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    # Try to get camera properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    fps = target_fps or cam_fps or 30

    os.makedirs(output_dir, exist_ok=True)


    # Initialize event camera
    input_resolution = (3, cam_height, cam_width)
    event_camera = cam_evb(threshold=initial_threshold,
                           input_resolution=input_resolution,
                           noise_level=initial_noise,
                           multi_threshold=multi_threshold)

    merge_idx = 0
    recording = False
    writer = None

    window_name = "EVB Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar parameters
    max_threshold_val = 2.0  # trackbar maps 0..1000 -> 0..max_threshold_val
    max_noise_val = 2.0      # trackbar maps 0..1000 -> 0..max_noise_val

    # create trackbars: threshold (0..1000), noise (0..1000)
    thr_init_pos = int(clamp(initial_threshold, 0.0, max_threshold_val) / max_threshold_val * 1000)
    cv2.createTrackbar('threshold', window_name, thr_init_pos, 1000, lambda x: None)
    noise_init_pos = int(clamp(initial_noise if initial_noise is not None else 0.0, 0.0, max_noise_val) / max_noise_val * 1000)
    cv2.createTrackbar('noise', window_name, noise_init_pos, 1000, lambda x: None)
    # spatial blur kernel size (0..31 -> odd sizes, 0 = no blur)
    cv2.createTrackbar('blur', window_name, 0, 31, lambda x: None)
    # temporal low-pass alpha (0..1000 -> 0.0..1.0) alpha close to 1 -> more smoothing
    cv2.createTrackbar('lp_alpha', window_name, 0, 1000, lambda x: None)

    # control state shared with mouse callback
    control_state = {
        'merge_idx': merge_idx,
        'recording': recording,
        'img_size': (cam_width, cam_height),
        'lp_prev': None
    }

    # mouse callback: toggle recording or change merge index based on click location
    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        w, h = control_state.get('img_size', (cam_width, cam_height))
        # button geometry
        btn_w = 70
        btn_h = 30
        spacing = 8
        x1 = w - 10 - (btn_w * 3 + spacing * 2)
        y1 = 10
        prev_rect = (x1, y1, x1 + btn_w, y1 + btn_h)
        rec_rect = (x1 + btn_w + spacing, y1, x1 + btn_w * 2 + spacing, y1 + btn_h)
        next_rect = (x1 + (btn_w + spacing) * 2, y1, x1 + (btn_w + spacing) * 2 + btn_w, y1 + btn_h)

        def inside(r, px, py):
            return (px >= r[0] and px <= r[2] and py >= r[1] and py <= r[3])

        if inside(prev_rect, x, y):
            # previous merge
            control_state['merge_idx'] = (control_state['merge_idx'] - 1) % len(merge_methods)
        elif inside(next_rect, x, y):
            control_state['merge_idx'] = (control_state['merge_idx'] + 1) % len(merge_methods)
        elif inside(rec_rect, x, y):
            control_state['recording'] = not control_state['recording']

    cv2.setMouseCallback(window_name, mouse_callback)

    print("Controls: q=quit, SPACE=toggle record, or click the on-screen buttons (Prev / REC / Next). Use sliders for threshold/noise.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Convert BGR (OpenCV) to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply spatial and temporal low-pass filtering based on trackbars
            blur_k = cv2.getTrackbarPos('blur', window_name)
            # ensure odd kernel size for Gaussian (0 => no blur)
            if blur_k > 0:
                if blur_k % 2 == 0:
                    blur_k += 1
                rgb = cv2.GaussianBlur(rgb, (blur_k, blur_k), 0)

            lp_alpha_pos = cv2.getTrackbarPos('lp_alpha', window_name)
            lp_alpha = float(lp_alpha_pos) / 1000.0
            if lp_alpha > 0.0:
                prev = control_state.get('lp_prev')
                if prev is None:
                    control_state['lp_prev'] = rgb.astype(np.float32)
                else:
                    # EMA: new = alpha*prev + (1-alpha)*current  (alpha closer to 1 => more smoothing)
                    prev = prev.astype(np.float32)
                    control_state['lp_prev'] = (lp_alpha * prev + (1.0 - lp_alpha) * rgb.astype(np.float32))
                rgb = np.clip(control_state['lp_prev'], 0, 255).astype(np.uint8)
            else:
                # reset previous when lp disabled
                control_state['lp_prev'] = None

            # Prepare tensor (C,H,W) float on correct device
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).to(event_camera.device).float()
            # Avoid zeros for log
            img_t[img_t <= 0] = 1e-6

            # Update event camera
            event_camera.update(img_t)

            # Get visualization as RGB image (H,W,3)
            vis = event_camera.out_rgb(color=merge_methods[merge_idx])

            # vis might be a torch tensor; convert to numpy uint8 BGR for OpenCV
            if isinstance(vis, torch.Tensor):
                vis_np = vis.detach().cpu().numpy()
                # If shape is (H,W,3) keep, if (C,H,W) adapt
                if vis_np.ndim == 3 and vis_np.shape[0] in (1, 2, 3):
                    # transpose to H,W,C
                    vis_np = np.transpose(vis_np, (1, 2, 0))
            else:
                vis_np = np.array(vis)

            # Ensure dtype and range
            vis_np = np.nan_to_num(vis_np)
            vis_np = np.clip(vis_np, 0, 255).astype(np.uint8)

            # If vis is RGB, convert to BGR for display
            if vis_np.shape[2] == 3:
                vis_display = cv2.cvtColor(vis_np, cv2.COLOR_RGB2BGR)
            else:
                vis_display = vis_np

            # Read trackbar positions for numeric parameters
            thr_pos = cv2.getTrackbarPos('threshold', window_name)
            noise_pos = cv2.getTrackbarPos('noise', window_name)

            # map trackbar values to real parameters
            event_camera.threshold = float(thr_pos) / 1000.0 * max_threshold_val
            noise_val = float(noise_pos) / 1000.0 * max_noise_val
            event_camera.noise = noise_val if noise_val > 0.0 else 0.0

            # Read merge and recording state from control_state
            merge_idx = control_state['merge_idx']
            recording = control_state['recording']

            # store current image size for mouse callback geometry
            control_state['img_size'] = (vis_display.shape[1], vis_display.shape[0])

            # Overlay status text
            status = f"merge={merge_methods[merge_idx]} thr={event_camera.threshold:.3f} noise={event_camera.noise:.3f} rec={'ON' if recording else 'OFF'}"
            cv2.putText(vis_display, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # draw On-screen buttons (Prev REC Next) at top-right
            w_vis = vis_display.shape[1]
            btn_w = 70
            btn_h = 30
            spacing = 8
            x1 = w_vis - 10 - (btn_w * 3 + spacing * 2)
            y1 = 10
            prev_rect = (x1, y1, x1 + btn_w, y1 + btn_h)
            rec_rect = (x1 + btn_w + spacing, y1, x1 + btn_w * 2 + spacing, y1 + btn_h)
            next_rect = (x1 + (btn_w + spacing) * 2, y1, x1 + (btn_w + spacing) * 2 + btn_w, y1 + btn_h)

            # draw rectangles
            cv2.rectangle(vis_display, (prev_rect[0], prev_rect[1]), (prev_rect[2], prev_rect[3]), (200, 200, 200), -1)
            # record button colored red when active
            rec_color = (0, 0, 255) if recording else (50, 200, 50)
            cv2.rectangle(vis_display, (rec_rect[0], rec_rect[1]), (rec_rect[2], rec_rect[3]), rec_color, -1)
            cv2.rectangle(vis_display, (next_rect[0], next_rect[1]), (next_rect[2], next_rect[3]), (200, 200, 200), -1)

            # text labels
            cv2.putText(vis_display, '<', (prev_rect[0] + 24, prev_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(vis_display, 'REC' if recording else 'REC', (rec_rect[0] + 10, rec_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_display, '>', (next_rect[0] + 26, next_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow(window_name, vis_display)

            # Handle recording
            if recording:
                if writer is None:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    out_name = os.path.join(output_dir, f"evb_record_{ts}_{merge_methods[merge_idx]}.mp4")
                    # construct fourcc manually to avoid static analysis issues
                    fourcc = (ord('m') | (ord('p') << 8) | (ord('4') << 16) | (ord('v') << 24))
                    writer = cv2.VideoWriter(out_name, fourcc, fps, (vis_display.shape[1], vis_display.shape[0]))
                    if not writer.isOpened():
                        print(f"Failed to open video writer for {out_name}")
                        writer = None
                if writer is not None:
                    writer.write(vis_display)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                # print(key)
                if key == ord('q'):
                    break
                elif key == 32:  # space
                    # toggle recording and sync the trackbar
                    recording = not recording
                    cv2.setTrackbarPos('record', window_name, int(recording))
                    if not recording and writer is not None:
                        writer.release()
                        writer = None
                    print(f"Recording {'started' if recording else 'stopped'}")
                elif key == ord('m'):
                    merge_idx = (merge_idx + 1) % len(merge_methods)
                    print(f"merge -> {merge_methods[merge_idx]}")
                elif key == ord('y'):
                    event_camera.threshold = float(event_camera.threshold) + 0.01
                    print(f"threshold -> {event_camera.threshold}")
                elif key == ord('r'):
                    event_camera.threshold = float(event_camera.threshold) - 0.01
                    event_camera.threshold = clamp(event_camera.threshold, 0.0, 10.0)
                    print(f"threshold -> {event_camera.threshold}")
                elif key == ord('n'):
                    if event_camera.noise is None:
                        event_camera.noise = 0.0
                    event_camera.noise = float(event_camera.noise) + 0.01
                    print(f"noise -> {event_camera.noise}")
                elif key == ord('b'):
                    if event_camera.noise is None:
                        event_camera.noise = 0.0
                    event_camera.noise = float(event_camera.noise) - 0.01
                    event_camera.noise = clamp(event_camera.noise, 0.0, 100.0)
                    print(f"noise -> {event_camera.noise}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live USB camera to event-based converter')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for cv2.VideoCapture')
    parser.add_argument('--output', type=str, default='./video_out', help='Output directory for recordings')
    parser.add_argument('--threshold', type=float, default=0.3, help='Initial threshold')
    parser.add_argument('--noise', type=float, default=0.8, help='Initial noise level')
    parser.add_argument('--fps', type=int, default=None, help='Target fps for recording (defaults to camera fps)')
    args = parser.parse_args()

    run_camera(camera_index=args.camera,
               output_dir=args.output,
               initial_threshold=args.threshold,
               initial_noise=args.noise,
               target_fps=args.fps)