import os

# Root directory containing all .npy files (flattened)
features_root = "data/aseel_custom/features/NPY FOR model learning"

# Output split file path
output_file = "data/aseel_custom/splits/train.split1.bundle"

# Collect all .npy video feature file paths (without .npy extension)
video_list = []

for fname in os.listdir(features_root):
    if fname.endswith(".npy"):
        video_name = os.path.splitext(fname)[0]
        
        # Skip multiples of 10 (10, 20, 30, ..., 100)
        try:
            num = int(video_name.replace("Video", ""))
            if num % 10 == 0:
                continue
        except ValueError:
            pass  # in case filename is not in 'VideoX' format

        video_list.append(video_name)

# Sort the list to keep consistent order
video_list = sorted(video_list, key=lambda x: int(x.replace("Video", "")))

# Write to split file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for video_name in video_list:
        f.write(video_name + "\n")

print(f"✅ Training split file created with {len(video_list)} videos at: {output_file}")
