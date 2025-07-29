import os

gt_dir = "data/aseel_custom/groundTruth"
split_file = "data/aseel_custom/splits/train.split1.bundle"

video_names = [
    fname.replace(".txt", "") for fname in os.listdir(gt_dir) if fname.endswith(".txt")
]

video_names.sort()  # Optional: keep order consistent

os.makedirs(os.path.dirname(split_file), exist_ok=True)

with open(split_file, "w") as f:
    for name in video_names:
        f.write(name + "\n")

print(f"✓ Created split file with {len(video_names)} videos.")
