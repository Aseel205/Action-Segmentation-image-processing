import cv2
import os

def extract_frames(video_path, output_dir, fps=3):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")
