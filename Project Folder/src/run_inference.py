import sys
import torch
import video_feature_utils
from model import SimpleTCN
import numpy as np
from scipy.stats import mode

# Label map must match training
label_map = {
    0: 'attach_1hand',
    1: 'background',
    2: 'plug_1hand',
    3: 'screwdriver_1hand',
    4: 'screwdriver_2hand'
}

FPS = 5
SMOOTHING_WINDOW = 7
MIN_SEGMENT_DURATION = 1.0

def smooth_predictions(preds, window_size=7):
    smoothed = []
    half = window_size // 2
    for i in range(len(preds)):
        start = max(0, i - half)
        end = min(len(preds), i + half + 1)
        smoothed.append(mode(preds[start:end], keepdims=False).mode)
    return np.array(smoothed)

def get_action_segments(preds, fps=FPS, min_duration=1.0):
    segments = []
    start_frame = 0
    current_label = preds[0]

    for i in range(1, len(preds)):
        if preds[i] != current_label:
            duration = (i - start_frame) / fps
            if duration >= min_duration and current_label != 1:  # skip background label=1
                segments.append((start_frame, i - 1, current_label))
            start_frame = i
            current_label = preds[i]

    # Final segment
    duration = (len(preds) - start_frame) / fps
    if duration >= min_duration and current_label != 1:
        segments.append((start_frame, len(preds) - 1, current_label))

    return [(s / fps, e / fps, label_map[l]) for s, e, l in segments]

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def main(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features from {video_path}...")
    features = video_feature_utils.extract_features_directly(video_path, fps=FPS, device=device)

    model = SimpleTCN(input_dim=features.shape[1], num_classes=len(label_map))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    features_tensor = torch.tensor(features.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        preds = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()

    preds = smooth_predictions(preds, window_size=SMOOTHING_WINDOW)
    segments = get_action_segments(preds)

    print("\nDetected action segments:")
    for start, end, action in segments:
        print(f"{format_time(start)} - {format_time(end)} : {action}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_inference.py <video_path>")
        sys.exit(1)
    video_path = "videos/Video_2.MP4"
    main(video_path)


## to activate  : python src/run_inference.py "vedios/Speaker Mounting Videos/Video_1.MP4"