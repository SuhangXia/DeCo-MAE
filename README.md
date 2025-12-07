
# DeCo-MAE: Decomposing Semantics for Compositional Zero-Shot Action Recognition

[](https://www.google.com/search?q=https://arxiv.org/abs/25XX.XXXXX)
[](https://opensource.org/licenses/MIT)
[](https://huggingface.co/LancetRobotics/DeCo-MAE)

This is the official PyTorch implementation of the paper: **"DeCo-MAE: Decomposing Semantics for Compositional Zero-Shot Action Recognition in Human-Robot Interaction"** (CVPR 2026 Submission).

We propose **DeCo-MAE**, a framework that enables a **VideoMAE V2 Giant** model to achieve **78.86% Zero-Shot accuracy** and **85.80% SOTA accuracy** on the HRI30 dataset by leveraging semantic decomposition and a novel cool-down training strategy.

-----

## 🏆 Model Zoo (Pre-trained Weights)

To verify our results immediately without re-training, you can download our trained checkpoints from [Hugging Face](https://huggingface.co/LancetRobotics/DeCo-MAE).

| Model Variant | Setting | Accuracy | Checkpoint |
| :--- | :--- | :---: | :--- |
| **DeCo-MAE (Final)** | Fully Supervised (Cool-down) | **85.80%** | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth) |
| **DeCo-MAE (Robust)** | Fully Supervised (Strong Aug) | 82.84% | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/final_sota_best.pth) |
| **DeCo-MAE (Zero-Shot)** | Zero-Shot Split (Unseen 5 classes) | **78.86%** | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/zeroshot_model.pth) |

> **Note:** Place downloaded `.pth` files into `./checkpoints_final/` or `./checkpoints_cooldown/` to skip training.

-----

## 🛠️ Environment Setup

This codebase was developed and tested on **Linux (Ubuntu 20.04)** with **Python 3.10+** and **PyTorch 2.4+**.
**Hardware Requirement:** NVIDIA GPU with at least **24GB VRAM** (RTX 3090/4090/5090) is recommended due to the 1B-parameter Giant backbone.

### 1\. Clone the repository

```bash
git clone https://github.com/SuhangXia/DeCo-MAE.git
cd DeCo-MAE
```

### 2\. Install dependencies

```bash
# If you are in China/AutoDL, enable acceleration first:
source /etc/network_turbo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers decord scikit-learn pandas matplotlib seaborn opencv-python-headless huggingface_hub accelerate
```

-----

## 📂 Data Preparation

Please organize the **HRI30** dataset structure as follows:

```
/root/hri30/
├── train/
│   ├── 1/
│   │   ├── CID01_SID01_VID01.avi
│   │   └── ...
│   ├── ...
│   └── 30/
├── train_set_labels.csv
└── ...
```

> **Note:** The code automatically splits data into training/validation or seen/unseen sets based on the experiment configuration.

-----

## 🚀 Reproduction Instructions (Training)

We provide a **3-stage training pipeline** to reproduce our results.

### Stage 1: Zero-Shot Verification (Ablation Study)

Train the model on 25 seen classes and evaluate on 5 unseen classes to verify semantic generalization.

```bash
python run_zeroshot_exp.py
```

  * **Expected Result:** Zero-Shot Accuracy ≈ **78.86%**
  * **Output:** `checkpoints_zeroshot/zeroshot_model.pth`

### Stage 2: Robustness Training (Strong Augmentation)

Train the full model on all 30 classes with Dual-Head and Strong Augmentation (30 Epochs).

```bash
python train_final_sota_long.py
```

  * **Expected Result:** Accuracy ≈ 82.84% (High robustness, but slight underfitting due to strong aug)
  * **Output:** `checkpoints_final/final_sota_best.pth`

### Stage 3: Cool-down Fine-tuning (The SOTA Step)

Fine-tune the Stage 2 model with weak augmentation and low learning rate (10 Epochs) to achieve optimal performance.

```bash
python train_cooldown.py
```

  * **Expected Result:** Accuracy ≈ **85.80%** (New SOTA)
  * **Output:** `checkpoints_cooldown/cooldown_best.pth`

-----

## ⚡ Evaluation & Visualization

### 1\. Evaluate SOTA Accuracy

To verify the final accuracy on the full dataset:

```bash
# Ensure you have 'cooldown_best.pth' in the correct folder
python eval_final.py
```

### 2\. Generate Qualitative Results (Attention Maps)

Visualize where the model is looking (Cross-Modal Attention):

```bash
python visualize_attention.py
```

  * **Output:** `attention_vis.png` (Shows heatmap focus on hands/tools)

### 3\. Generate Paper Figures

Generate the Confusion Matrix and Training Dynamics plots used in the paper:

```bash
python generate_paper_plots.py
```

  * **Output:** `fig_confusion_matrix.png`, `fig_training_dynamics.png`

-----

## 📝 Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{xia2026decomae,
  title={DeCo-MAE: Decomposing Semantics for Compositional Zero-Shot Action Recognition in Human-Robot Interaction},
  author={Xia, Suhang},
  journal={CVPR Submission},
  year={2026}
}
```

## 📧 Contact

For any questions, please contact Suhang Xia at `suhang.xia@kcl.ac.uk`.
