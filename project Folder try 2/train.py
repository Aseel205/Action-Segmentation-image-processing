import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SimpleTCN

class ActionSegDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # shape: (num_frames, feature_dim)
        self.labels = np.load(labels_path)      # shape: (num_frames, )

    def __len__(self):
        return 1  # One video sequence per dataset (for now)

    def __getitem__(self, idx):
        # Return entire video sequence (features and labels)
        feature_seq = torch.tensor(self.features.T, dtype=torch.float32)  # (feature_dim, seq_len)
        label_seq = torch.tensor(self.labels, dtype=torch.long)           # (seq_len,)
        return feature_seq, label_seq

def train():
    dataset = ActionSegDataset("features/demo.npy", "labels/demo_labels.npy")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_dim = dataset.features.shape[1]
    num_classes = len(set(dataset.labels))
    model = SimpleTCN(input_dim=feature_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    model.train()
    for epoch in range(epochs):
        for features, labels in dataloader:
            features = features.to(device)  # (batch=1, feature_dim, seq_len)
            labels = labels.to(device)      # (batch=1, seq_len)
            
            optimizer.zero_grad()
            outputs = model(features)       # (batch=1, seq_len, num_classes)
            outputs = outputs.view(-1, num_classes)  # (seq_len, num_classes)
            labels = labels.view(-1)                 # (seq_len,)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
