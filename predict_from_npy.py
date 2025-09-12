import os
import torch
import numpy as np
from model import Trainer

# --- CONFIG ---
epoch_to_load = 50

# Single video to predict
video_name = "Video80.npy"  # just the filename
features_folder = "data/aseel_custom/features/NPY FOR model learning"  # folder containing all .npy
model_dir = "./models/aseel_custom/split_1"
results_dir = "results/NPY_for_model_learning"  # output folder
mapping_file = "./data/aseel_custom/mapping.txt"

os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load action label mapping ---
actions_dict = {}
with open(mapping_file, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) >= 2:
            actions_dict[tokens[1]] = int(tokens[0])
idx_to_action = {v: k for k, v in actions_dict.items()}
num_classes = len(actions_dict)

# --- Load features ---
feature_file = os.path.join(features_folder, video_name)
if not os.path.exists(feature_file):
    raise FileNotFoundError(f"Feature file not found: {feature_file}")

features = np.load(feature_file)
input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)  # (1, 2048, T)

# --- Load model ---
trainer = Trainer(num_blocks=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=num_classes)
trainer.model.load_state_dict(torch.load(f"{model_dir}/epoch-{epoch_to_load}.model", map_location=device))
trainer.model.to(device)
trainer.model.eval()

# --- Predict ---
with torch.no_grad():
    predictions = trainer.model(input_x)
    _, predicted = torch.max(predictions[-1].data, 1)
    predicted = predicted.squeeze().cpu().numpy()

predicted_labels = [idx_to_action[idx] for idx in predicted]

# --- Compress consecutive duplicates ---
def compress_predictions(labels):
    if not labels:
        return []
    compressed = [labels[0]]
    for label in labels[1:]:
        if label != compressed[-1]:
            compressed.append(label)
    return compressed

compressed_labels = compress_predictions(predicted_labels)

# --- Save predictions ---
output_file = os.path.join(results_dir, video_name.replace(".npy", ".txt"))
with open(output_file, "w") as f:
    for label in compressed_labels:
        f.write(label + "\n")

print(f"Prediction saved to: {output_file}")
print("Compressed predicted actions:")
for label in compressed_labels:
    print(label)
