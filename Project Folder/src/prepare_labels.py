import numpy as np
import re

def parse_time(t):
    """Convert a timestamp like '1:25' to seconds (e.g., 85)."""
    minutes, seconds = map(int, t.strip().split(":"))
    return minutes * 60 + seconds

def parse_annotations(annot_path, num_frames, fps):
    frame_labels = ["background"] * num_frames

    with open(annot_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract timestamps (with optional spaces around '-')
            times = re.findall(r'\((\d+:\d+)\s*-\s*(\d+:\d+)\)', line)
            if not times:
                continue
            start_time, end_time = map(parse_time, times[0])

            # Calculate frame indices
            start_idx = int(start_time * fps)
            end_idx = int(end_time * fps)

            # Extract label words (everything before '(')
            label_text = line.split('(')[0].strip()

            # Remove duplicate words and extra spaces
            words = label_text.split()
            # Remove duplicates (keeping order)
            seen = set()
            filtered_words = []
            for w in words:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    filtered_words.append(w)

            # Join last two words as label (e.g., "ScrewDriver 1hand" or "attach 1hand")
            if len(filtered_words) >= 2:
                label = filtered_words[-2] + "_" + filtered_words[-1]
            else:
                label = filtered_words[-1]

            label = label.lower()  # lowercase for consistency

            # Assign label to frames
            for i in range(start_idx, min(end_idx + 1, num_frames)):
                frame_labels[i] = label

    return frame_labels

def save_frame_labels(annot_path, feature_path, output_path, fps):
    features = np.load(feature_path)
    num_frames = len(features)

    frame_labels = parse_annotations(annot_path, num_frames, fps)
    label_set = sorted(set(frame_labels))
    label_to_index = {label: i for i, label in enumerate(label_set)}
    label_indices = [label_to_index[label] for label in frame_labels]

    np.save(output_path, np.array(label_indices))
    print(f"Saved frame labels to {output_path}")
    print(f"Label map: {label_to_index}")


if __name__ == "__main__":
    annot_path = "data/Speaker Mounting Video_1.txt"     # your annotation file
    feature_path = "features/Video_1.npy"                # features from previous step
    output_path = "labels/Video_1_labels.npy"            # output label file
    fps = 5                                              # same FPS used for frames & features
    save_frame_labels(annot_path, feature_path, output_path, fps)
