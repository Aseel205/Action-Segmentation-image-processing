import os
import shutil

# Root folder containing your 4 category folders
root_dir = r"C:\Users\aseel\OneDrive\Desktop\נושאים בעיבוד תמונה\ms-tcn-master\Original"

# Output flat folder
flat_dir = os.path.join(root_dir, "AllLabels")
os.makedirs(flat_dir, exist_ok=True)

# The 4 folders in the exact order you want
folders = ["fan_mounting", "fan_unmounting", "speaker_mounting", "speaker_unmounting"]

counter = 1

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # Repeat each folder 3 times
    for repeat in range(3):
        for file in txt_files:
            old_path = os.path.join(folder_path, file)
            new_name = f"Video{counter}.txt"
            new_path = os.path.join(flat_dir, new_name)

            shutil.copy2(old_path, new_path)
            print(f"Copied: {old_path} -> {new_path}")

            counter += 1

print(f"✅ Created {counter-1} .txt files in: {flat_dir}")
