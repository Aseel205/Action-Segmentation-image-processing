import os
import re

# Input + output
groundtruth_dir = r"data\aseel_custom\groundTruth"
output_dir = r"data\aseel_custom\groundTruth\FinalData"
fps = 5  # adjust if needed

os.makedirs(output_dir, exist_ok=True)

# --- Utilities ---
def timestamp_to_seconds(t):
    parts = t.split(":")
    parts = list(map(int, parts))
    if len(parts) == 2:  # mm:ss
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:  # hh:mm:ss
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def parse_label_line(line):
    m = re.match(r"^(.*?)\s*\(\s*(\d+(?::\d+){1,2})\s*-\s*(\d+(?::\d+){1,2})\s*\)", line.strip())
    if not m:
        return None, None, None
    label = m.group(1).strip()
    start_sec = timestamp_to_seconds(m.group(2))
    end_sec = timestamp_to_seconds(m.group(3))
    return label, start_sec, end_sec

def convert_label_file(filepath, output_path):
    print(f"Processing {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    frame_labels = []
    for line in lines:
        label, start_sec, end_sec = parse_label_line(line)
        if label is None:
            print(f"⚠️ Skipping bad line: {line.strip()}")
            continue
        num_frames = int(round((end_sec - start_sec) * fps))
        if num_frames > 0:
            frame_labels.extend([label] * num_frames)

    with open(output_path, "w", encoding="utf-8") as f:
        for lbl in frame_labels:
            f.write(lbl + "\n")

    print(f"✅ Saved: {output_path}")

# --- Main loop ---
for fname in sorted(os.listdir(groundtruth_dir)):
    if fname.endswith(".txt"):
        input_path = os.path.join(groundtruth_dir, fname)
        output_path = os.path.join(output_dir, fname)
        convert_label_file(input_path, output_path)

print("🎉 Done parsing all txt files")
