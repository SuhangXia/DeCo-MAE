# DeCo-MAE: Decomposing Semantics for Compositional Zero-Shot Action Recognition

[](https://www.google.com/search?q=https://arxiv.org/abs/25XX.XXXXX)
[](https://huggingface.co/LancetRobotics/DeCo-MAE)
[](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the **7CCEMSAP Final Project**.

We propose a **Dual-Head VideoMAE V2 Giant** framework that leverages **Semantic Decomposition** and a novel **Cool-down Training Strategy**. Our method achieves state-of-the-art performance on the HRI30 dataset, significantly outperforming standard baselines in both supervised and zero-shot settings.

## 🏆 Key Results

| Model Variant | Setting | Accuracy | Checkpoint |
| :--- | :--- | :---: | :--- |
| **DeCo-MAE (Final)** | **Fully Supervised (SOTA)** | **85.80%** | [Download](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth) |
| **DeCo-MAE (Robust)** | Strong Augmentation | 82.84% | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/final_sota_best.pth) |
| **DeCo-MAE (Zero-Shot)** | Zero-Shot Split (5 Unseen) | **78.86%** | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/zeroshot_model.pth) |

-----

## 🛠️ Environment Setup

This codebase is optimized for **Linux (Ubuntu 20.04)** with **NVIDIA GPUs** (24GB+ VRAM recommended, e.g., RTX 3090/4090).

```bash
# 1. Clone the repository
git clone https://github.com/SuhangXia/DeCo-MAE.git
cd DeCo-MAE

# 2. Install dependencies
# (Optional) Create a conda environment
conda create -n decomae python=3.10 -y
conda activate decomae

# Install PyTorch and required libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers decord scikit-learn pandas matplotlib seaborn opencv-python-headless huggingface_hub accelerate
```

-----

## 📂 Data Preparation

Please organize the **HRI30** dataset as follows. The code will automatically handle train/test splits.

```
/root/hri30/
├── train/                  # Training videos
│   ├── 1/
│   ├── ...
│   └── 30/
├── test_set/               # Test videos (for submission)
│   ├── CIDxx_SIDxx_VIDxx.avi
│   └── ...
└── train_set_labels.csv    # Training labels
```

-----

## 🚀 Quick Start: Generate Submission (For Coursework)

To generate the `test_set_labels.csv` required for the final project submission using our best model (**85.80% accuracy**):

1.  **Download the pre-trained weight**:
    Download `cooldown_best.pth` from [Hugging Face](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth) and place it in `./checkpoints_cooldown/`.

    ```bash
    mkdir -p checkpoints_cooldown
    wget https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth -O checkpoints_cooldown/cooldown_best.pth
    ```

2.  **Run Inference**:

    ```bash
    python inference.py
    ```

3.  **Output**:
    The file `test_set_labels.csv` will be generated in the root directory.

-----

## 🔬 Reproducing Training (Step-by-Step)

We provide a **3-stage training pipeline** to fully reproduce our results.

### Stage 1: Zero-Shot Verification (Ablation Study)

Verifies the semantic generalization capability by training on 25 classes and testing on 5 unseen classes.

```bash
python train_stage1_zeroshot.py
```

  * **Goal**: Achieve \~78.86% Zero-Shot Accuracy.

### Stage 2: Robustness Training (Strong Augmentation)

Trains the full model on all 30 classes using Dual-Head architecture and strong data augmentations (RandomResizedCrop, Flip, etc.).

```bash
python train_stage2_robust.py
```

  * **Goal**: Learn robust features (Acc \~82.84%).
  * **Output**: `checkpoints_final/final_sota_best.pth`

### Stage 3: Cool-down Fine-tuning (The SOTA Step)

The critical step to achieve SOTA. We fine-tune the robust model with weak augmentation and a low learning rate to bridge the distribution gap.

```bash
python train_stage3_cooldown.py
```

  * **Goal**: Achieve **85.80%** Final Accuracy.
  * **Output**: `checkpoints_cooldown/cooldown_best.pth`

-----

## 📊 Visualization & Analysis

### 1\. Generate Paper Figures

Generate the **Confusion Matrix** and **Training Dynamics** plots used in the paper/report.

```bash
python plot_figures.py
```

  * **Output**: `fig_confusion_matrix.png`, `fig_training_dynamics.png`

### 2\. Visualize Attention Maps

Generate heatmaps to see where the model focuses (e.g., hands and tools).

```bash
python visualize.py
```

  * **Output**: `attention_vis.png`

### 3\. Evaluate Accuracy

To manually evaluate the accuracy on the validation set:

```bash
python evaluate.py
```


## 📧 Contact

**Suhang Xia** King's College London  
Email: `suhang.xia@kcl.ac.uk`
