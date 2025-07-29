#!/usr/bin/env python3

import torch
import os
import argparse
import random
from model import Trainer
from batch_gen import BatchGenerator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix seeds for reproducibility
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', choices=['train', 'predict'])
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
args = parser.parse_args()

# Hyperparameters
num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
batch_size = 1
learning_rate = 0.0005
num_epochs = 50

# Adjust sample rate for specific datasets
sample_rate = 2 if args.dataset == "50salads" else 1

# Paths
base_path = f"./data/{args.dataset}"
vid_list_file = f"{base_path}/splits/train.split{args.split}.bundle"
vid_list_file_tst = f"{base_path}/splits/test.split{args.split}.bundle"
features_path = f"{base_path}/features/"
gt_path = f"{base_path}/groundTruth/"
mapping_file = f"{base_path}/mapping.txt"

model_dir = f"./models/{args.dataset}/split_{args.split}"
results_dir = f"./results/{args.dataset}/split_{args.split}"

# Create output directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Read action mappings safely
actions_dict = {}
with open(mapping_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            actions_dict[parts[1]] = int(parts[0])

num_classes = len(actions_dict)

# Trainer
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)

# Run action
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(
        model_dir,
        batch_gen,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

elif args.action == "predict":
    trainer.predict(
        model_dir,
        results_dir,
        features_path,
        vid_list_file_tst,
        num_epochs,
        actions_dict,
        device,
        sample_rate
    )
