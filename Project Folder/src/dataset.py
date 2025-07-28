# src/dataset.py
import glob, os, numpy as np, torch
from torch.utils.data import Dataset

class MultiVideoDataset(Dataset):
    def __init__(self, features_dir, labels_dir, video_ids):
        """video_ids = list like ['Video_1', 'Video_2', ...]"""
        self.fpaths   = [os.path.join(features_dir, f"{vid}.npy") for vid in video_ids]
        self.lpaths   = [os.path.join(labels_dir,   f"{vid}_labels.npy") for vid in video_ids]

        # Pre‑load meta info so we can build the model once
        sample = np.load(self.fpaths[0])
        self.feature_dim = sample.shape[1]

        # Build global label map across *all* videos
        all_labels = set()
        for p in self.lpaths:
            all_labels.update(np.load(p))
        self.num_classes = len(all_labels)

    def __len__(self):
        return len(self.fpaths)          # 5 videos → 5
    def __getitem__(self, idx):
        feats  = torch.tensor(np.load(self.fpaths[idx]).T, dtype=torch.float32)  # (C, T)
        labels = torch.tensor(np.load(self.lpaths[idx]),  dtype=torch.long)      # (T,)
        return feats, labels
