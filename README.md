# DeCo-MAE: Decomposing Semantics for Compositional Zero-Shot Action Recognition

**[7CCEMSAP Final Project]** | **Suhang Xia, Ruiyi Hu, Muye Yuan** (King's College London)

[](https://www.google.com/search?q=https://arxiv.org/abs/25XX.XXXXX)
[](https://huggingface.co/LancetRobotics/DeCo-MAE)
[](https://opensource.org/licenses/MIT)

We propose a **Dual-Head VideoMAE V2 Giant** framework that achieves **85.80% SOTA** and **78.86% Zero-Shot** accuracy on the HRI30 dataset. Our method relies on **Semantic Decomposition** and a novel **Cool-down Training Strategy**.

## 🏆 Key Results

| Model Variant | Setting | Accuracy | Checkpoint |
| :--- | :--- | :---: | :--- |
| **DeCo-MAE (Final)** | **Fully Supervised (SOTA)** | **85.80%** | [Download](https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth) |
| **DeCo-MAE (Robust)** | Strong Augmentation | 82.84% | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/final_sota_best.pth) |
| **DeCo-MAE (Zero-Shot)** | Zero-Shot Split (5 Unseen) | **78.86%** | [Download](https://www.google.com/search?q=https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/zeroshot_model.pth) |

-----

## 🛠️ Environment Setup

This codebase is optimized for **Linux** with **Python 3.10+**.

### ⚠️ Hardware Requirement (Crucial for Reproduction)

Due to the size of the **VideoMAE V2 Giant** backbone (over 1 billion parameters), a high-VRAM GPU is required for training:

  * **Minimum VRAM:** 24GB
  * **Recommended VRAM:** **32GB+** (e.g., **NVIDIA RTX 5090** or higher)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/SuhangXia/DeCo-MAE.git
cd DeCo-MAE

# 2. Install dependencies (using conda or venv)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers decord scikit-learn pandas matplotlib seaborn opencv-python-headless huggingface_hub accelerate
```

-----

## 📂 Data Preparation

The code expects the **HRI30** dataset to be structured by Class ID (1-30).

```
/root/hri30/
├── train/                  # ✅ Structured training videos (Class ID folders)
├── train_set/              # train original vedio (Source: All videos mixed)
├── test_set/               # Raw test videos
└── train_set_labels.csv    # Labels mapping filenames to IDs
```

### 1\. Structure Raw Data (If Needed)

If your original videos are currently mixed in the `train_set/` folder, run the **`organize_data.py`** script to automatically classify and move them into the required `train/1`, `train/2`, etc., structure:

```bash
python organize_data.py
```

-----

## 🚀 Quick Start: Generate Submission (`test_set_labels.csv`)

This is the required final step for the 7CCEMSAP Coursework. Follow these steps to generate the final CSV using our best model:

1.  **Download the SOTA Weights**:

    ```bash
    mkdir -p checkpoints_cooldown
    wget https://huggingface.co/LancetRobotics/DeCo-MAE/resolve/main/cooldown_best.pth -O checkpoints_cooldown/cooldown_best.pth
    ```

2.  **Run Inference**:
    This script generates the final CSV file using the **85.80%** model.

    ```bash
    python inference.py
    ```

3.  **Output**:
    The file `test_set_labels.csv` is generated in the root directory.

-----

## 🔬 Reproducing Training (3-Stage Pipeline)

To fully verify the paper's ablation studies and final SOTA result:

### Stage 1: Zero-Shot Verification

```bash
python train_stage1_zeroshot.py
```

  * **Verifies**: Semantic generalization.

### Stage 2: Robustness Training (Strong Augmentation)

```bash
python train_stage2_robust.py
```

  * **Goal**: Learn robust features (Acc $\sim$82.84%).

### Stage 3: Cool-down Fine-tuning

```bash
python train_stage3_cooldown.py
```

  * **Goal**: Achieve **85.80%** Final Accuracy.

-----

## 📜 Team Contributions

| Member | Role & Primary Contributions |
| :--- | :--- |
| **Suhang Xia** | Lead Developer, Core Algorithm Design (DeCo-MAE Architecture), Paper Drafting. |
| **Ruiyi Hu** | Reproducibility Testing, Data Augmentation Design and Implementation Strategy. |
| **Muye Yuan** | Poster Design and Final Presentation Assistance. |

-----


## 📧 Contact

**Suhang Xia** King's College London  
Email: `suhang.xia@kcl.ac.uk`
