import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from model import SimpleTCN

FPS = 3  # Same as training

# Step 1: Extract frames from video
def extract_frames(video_path, output_dir, fps=FPS):
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
    return saved

# Step 2: Extract features using ResNet18
def extract_features(frames_dir):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    features = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    with torch.no_grad():
        for fname in frame_files:
            img = Image.open(os.path.join(frames_dir, fname)).convert('RGB')
            input_tensor = transform(img).unsqueeze(0)
            feat = model(input_tensor).squeeze().numpy()
            features.append(feat)
    features = np.stack(features)
    return features

# Step 3: Load model and predict actions
def predict_actions(features, model_path, device):
    features_tensor = torch.tensor(features.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, feat_dim, seq_len)
    model = SimpleTCN(input_dim=features.shape[1], num_classes=5)  # num_classes = your labels count
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)  # (1, seq_len, num_classes)
        preds = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()  # (seq_len,)
    return preds

# Step 4: Map indices back to labels
label_map = {0: 'attach_1hand', 1: 'background', 2: 'plug_1hand', 3: 'screwdriver_1hand', 4: 'screwdriver_2hand'}

# Step 5: Convert frame-wise predictions to action segments
def get_action_segments(preds, fps=3, min_duration=0.5):
    segments = []
    start_frame = 0
    current_label = preds[0]

    for i in range(1, len(preds)):
        if preds[i] != current_label:
            duration = (i - start_frame) / fps
            if duration >= min_duration and current_label != label_map_inv['background']:
                segments.append((start_frame, i - 1, current_label))
            start_frame = i
            current_label = preds[i]

    # Last segment
    duration = (len(preds) - start_frame) / fps
    if duration >= min_duration and current_label != label_map_inv['background']:
        segments.append((start_frame, len(preds) - 1, current_label))

    # Merge adjacent segments with the same label if they are close
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev_start, prev_end, prev_label = merged_segments[-1]
            curr_start, curr_end, curr_label = seg
            # If same label and gap <= 1 frame, merge
            if curr_label == prev_label and curr_start - prev_end <= 1:
                merged_segments[-1] = (prev_start, curr_end, prev_label)
            else:
                merged_segments.append(seg)

    # Convert frames to times
    final_segments = []
    for start_f, end_f, label_idx in merged_segments:
        start_time = start_f / fps
        end_time = end_f / fps
        final_segments.append((start_time, end_time, label_map[label_idx]))

    return final_segments


label_map_inv = {v:k for k,v in label_map.items()}

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main(video_path):
    frames_dir = "temp_frames"
    print(f"Extracting frames from {video_path}...")
    num_frames = extract_frames(video_path, frames_dir, fps=FPS)
    print(f"Extracted {num_frames} frames.")

    print("Extracting features...")
    features = extract_features(frames_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting actions on device {device}...")
    preds = predict_actions(features, "model.pth", device)

    print("Post-processing predictions into segments...")
    segments = get_action_segments(preds, fps=FPS)

    print("\nDetected Action Segments:")
    for start, end, action in segments:
        print(f"{format_time(start)} - {format_time(end)} : {action}")

    # Clean up temp frames
    import shutil
    shutil.rmtree(frames_dir)


if __name__ == "__main__":
   
    main("videos/Video_2.MP4")
