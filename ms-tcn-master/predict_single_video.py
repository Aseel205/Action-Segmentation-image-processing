import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from model import Trainer

# --- CONFIG ---
dataset = "gtea"
split = "1"
epoch_to_load = 50
target_fps = 5  # FPS for feature extraction

video_name = "fan_mounting_1.MP4"
video_dir = "./data/aseel_custom/videos/"

# Paths aligned with the working script
features_path = "./data/aseel_custom/features/"
model_dir = "./models/aseel_custom/split_1"
results_dir = f"./results/{dataset}/split_{split}"
mapping_file = "./data/aseel_custom/mapping.txt"

os.makedirs(features_path, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# --- Step 1: Extract Features from Video using ResNet50 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet50 without FC layer
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features_from_video(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(orig_fps / target_fps))

    features = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = resnet(input_tensor).squeeze().cpu().numpy()  # (2048,)

            features.append(feat)

        frame_idx += 1

    cap.release()

    if len(features) == 0:
        raise RuntimeError("No frames extracted.")

    features = np.stack(features, axis=1)  # (2048, num_frames)
    return features

# Extract and save features
video_path = os.path.join(video_dir, video_name)
print(f"Extracting features from video: {video_path}")
features = extract_features_from_video(video_path, target_fps)
feature_file = os.path.join(features_path, os.path.splitext(video_name)[0] + ".npy")
np.save(feature_file, features)
print(f"Features saved to: {feature_file}")

# --- Step 2: Load model and predict ---

# Load action label mapping
actions_dict = {}
with open(mapping_file, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) >= 2:
            actions_dict[tokens[1]] = int(tokens[0])
idx_to_action = {v: k for k, v in actions_dict.items()}
num_classes = len(actions_dict)

# Load features for prediction
features = np.load(feature_file)
input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)  # (1, 2048, T)

trainer = Trainer(num_blocks=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=num_classes)
trainer.model.load_state_dict(torch.load(f"{model_dir}/epoch-{epoch_to_load}.model"))
trainer.model.to(device)
trainer.model.eval()

with torch.no_grad():
    predictions = trainer.model(input_x)
    _, predicted = torch.max(predictions[-1].data, 1)
    predicted = predicted.squeeze().cpu().numpy()

predicted_labels = [idx_to_action[idx] for idx in predicted]

# Save predictions
output_file = os.path.join(results_dir, os.path.splitext(video_name)[0])
with open(output_file, "w") as f:
    f.write("### Frame level recognition: ###\n")
    f.write(" ".join(predicted_labels))

print(f"Prediction saved to: {output_file}")
print("Predicted actions:", predicted_labels)
