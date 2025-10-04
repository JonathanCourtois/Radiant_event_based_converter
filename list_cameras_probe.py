# list_cameras_probe.py
import cv2

def probe_cameras(max_index=10, test_frame=True):
    available = []
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        info = {"index": i}
        # try to read a frame to confirm functionality
        if test_frame:
            ret, frame = cap.read()
            info["works"] = bool(ret)
            if ret and frame is not None:
                info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        else:
            info["works"] = True

        available.append(info)
        cap.release()
    return available

if __name__ == "__main__":
    cams = probe_cameras(max_index=10)
    if not cams:
        print("No cameras found (tried indices 0..10).")
    else:
        print("Found cameras:")
        for c in cams:
            print(f" index={c['index']} works={c['works']} size={c.get('width','?')}x{c.get('height','?')} fps={c.get('fps','?')}")