import os 
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Change these paths to your setup
VIDEO_DIR = './data/aseel_custom/videos/Fan Mounting/'     # Your 6 raw videos folder
FEATURES_DIR = './data/aseel_custom/features/fan_mounting/' # Output folder for .npy features

os.makedirs(FEATURES_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet50 model without classification head
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
model.to(device)
model.eval()

# Image preprocessing pipeline matching ResNet50 training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    features = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / target_fps))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting features from {os.path.basename(video_path)} ({frame_count} frames, every {frame_interval}th frame)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(input_tensor)  # Output shape: [1, 2048, 1, 1]
                feat = feat.squeeze().cpu().numpy()  # Shape: (2048,)

            features.append(feat)

        frame_idx += 1

    cap.release()

    if len(features) == 0:
        print(f"Warning: No frames found in {video_path}")
        return None

    features = np.stack(features, axis=1)  # Shape: (2048, num_frames)
    return features

def main():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        feats = extract_features(video_path, target_fps=5)
        if feats is None:
            continue
        feature_path = os.path.join(FEATURES_DIR, os.path.splitext(video_file)[0] + '.npy')
        np.save(feature_path, feats)
        print(f"Saved features: {feature_path}")

if __name__ == '__main__':
    main()
