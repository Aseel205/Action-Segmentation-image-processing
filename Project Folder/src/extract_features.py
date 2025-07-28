import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Load pretrained ResNet18 (we remove the final classification layer)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features_from_folder(frames_dir, output_path):
    features = []
    frame_files = sorted(os.listdir(frames_dir))

    for fname in frame_files:
        if fname.endswith(".jpg"):
            path = os.path.join(frames_dir, fname)
            img = Image.open(path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                feat = model(input_tensor).squeeze().numpy()  # Shape: (512,)
            features.append(feat)

    features = np.stack(features)  # Shape: (num_frames, 512)
    np.save(output_path, features)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    frames_dir = "frames\\video1"         # folder with extracted frames
    output_path = "features\\Video_1.npy"  # where to save features
    extract_features_from_folder(frames_dir, output_path)
