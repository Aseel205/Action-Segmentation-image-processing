# Action Segmentation with MS-TCN++

This project adapts the **MS-TCN++** architecture from the paper  
[*MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation*](https://arxiv.org/abs/2006.09220)  
to segment and classify actions in **custom workflow videos** (e.g., fan mounting, speaker mounting, etc.).

We take the original research implementation and **apply it to our own dataset** by:
- Preparing custom **ground truth labels** for each video.
- Converting raw video features into `.npy` format for training.
- Modifying the training and evaluation scripts to work with our dataset.
- Using MS-TCN++ to reduce over-segmentation errors and improve segment boundaries.

---

## 📂 Project Structure

| File / Folder | Description |
|---------------|-------------|
| **main.py** | Main entry point for training and prediction. Handles arguments (`--action`, `--dataset`, `--split`) and calls the model. |
| **model.py** | Implementation of the MS-TCN/MS-TCN++ model — defines prediction generation stage, refinement stages, and loss functions. |
| **batch_gen.py** | Data loader — reads `.npy` features and ground truth `.txt` files, produces batches for training/testing. |
| **makingNPY.py** | Converts extracted video features into `.npy` format required by the model. |
| **create_split_file.py** | Generates train/test split files for the dataset. |
| **predict_from_npy.py** | Runs prediction when features are already in `.npy` format. |
| **predict_single_video.py** | Predicts action segments for a single input video feature file. |
| **eval.py** | Evaluates model predictions against ground truth using accuracy, F1 score, and edit score. |
| **script.py** | Utility script for running automated experiments or custom preprocessing (project-specific). |
| **data/** | Contains dataset definition: mapping files, ground truth labels, and (in practice) `.npy` feature files. |
| **data/aseel_custom/mapping.txt** | Maps numeric class IDs to human-readable action labels. |
| **data/aseel_custom/groundTruth/** | Ground truth action label sequences for each frame in each video, grouped by task (fan_mounting, speaker_mounting, etc.). |
| **LICENSE** | License file from the original repository. |

---

## 🚀 How to Run

### 1. Prepare Data
1. Place your `.npy` feature files in:
   ```
   data/aseel_custom/features/<category>/
   ```
   with a `test/` subfolder for evaluation videos.
2. Place corresponding ground truth label files (`.txt`) in:
   ```
   data/aseel_custom/groundTruth/<category>/
   ```
3. Update `mapping.txt` with class labels.

---

### 2. Train the Model
```bash
python main.py --action train --dataset aseel_custom --split 1
```
- `--action train` → runs training
- `--dataset aseel_custom` → our custom dataset folder name
- `--split` → train/test split number (use `create_split_file.py` to define)

---

### 3. Predict on Test Set
```bash
python main.py --action predict --dataset aseel_custom --split 1
```

---

### 4. Evaluate
```bash
python eval.py --dataset aseel_custom --split 1
```
Outputs:
- **Frame-wise Accuracy**
- **Segmental F1@10, F1@25, F1@50**
- **Edit Score**

---

## 🧠 Model Overview
- **MS-TCN++** is a multi-stage temporal convolutional network.
- First stage: generates coarse prediction from features.
- Refinement stages: iteratively smooth predictions and fix boundaries.
- Uses **dual dilated layers** to capture both short-term and long-term temporal context.
- Loss = classification (cross-entropy) + smoothing loss (to reduce over-segmentation).

---

## 📜 Reference
If you use this code, please cite:

```bibtex
@inproceedings{li2020ms,
  title={MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation},
  author={Li, Shijie and Abu Farha, Yazan and Liu, Yun and Cheng, Ming-Ming and Gall, Juergen},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```

---
