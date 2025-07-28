# src/train_multi.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MultiVideoDataset
from model import SimpleTCN

def train(video_ids, epochs=20, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Corrected features path (you had "vedios" but your folder is "vedios/Speaker Mounting Videos")
    features_dir = "features"
    labels_dir = "labels"
    
    ds = MultiVideoDataset(features_dir, labels_dir, video_ids)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    model = SimpleTCN(ds.feature_dim, ds.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for feats, labs in dl:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(feats)  # (batch=1, seq_len, num_classes)
            loss = criterion(outputs.view(-1, ds.num_classes), labs.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch:02d}/{epochs} — Loss: {running_loss / len(dl):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✓ model.pth saved")

if __name__ == "__main__":
    vids = ["Video_1"]  # Add or remove as needed
    train(vids)
