import cv2
import torch
import numpy as np
from torchvision import models, transforms

def extract_features_directly(video_path, fps=5, batch_size=16, device='cpu'):
    # Load pretrained ResNet18 and remove final FC layer
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),             # Convert numpy array directly to PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    batch_frames = []
    features = []
    count = 0
    kept = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                # frame is BGR numpy array; convert to RGB inside transform
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Apply transform
                input_tensor = transform(frame_rgb)
                batch_frames.append(input_tensor)

                if len(batch_frames) == batch_size:
                    batch_tensor = torch.stack(batch_frames).to(device)
                    feats = resnet(batch_tensor).squeeze(-1).squeeze(-1)  # (batch_size, 512)
                    features.append(feats.cpu().numpy())
                    batch_frames = []
                kept += 1

            count += 1

        # Process leftover frames
        if batch_frames:
            batch_tensor = torch.stack(batch_frames).to(device)
            feats = resnet(batch_tensor).squeeze(-1).squeeze(-1)
            features.append(feats.cpu().numpy())

    cap.release()
    print(f"Extracted {kept} feature vectors (no frames saved).")
    return np.vstack(features)
