import os
# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ================= 配置 =================
MODEL_ID = "OpenGVLab/VideoMAEv2-giant" 
CHECKPOINT_PATH = "/root/autodl-tmp/checkpoints/sota_v2_best.pth"

NUM_FRAMES = 16      
IMG_SIZE = 224
BATCH_SIZE = 8       # 推理不占显存，开大点没事
CACHE_DIR = "/root/autodl-tmp/hf_cache"

print(f"🚀 开始评估 Baseline | 模型: V2 Giant | 权重: {CHECKPOINT_PATH}")

# ================= 模型定义 =================
class VideoMAEv2_Classifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading Architecture: {model_id} ...")
        
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        config.use_cache = False 
        
        self.backbone = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True,
            config=config,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16
        )
        
        if hasattr(config, "hidden_size"):
            self.hidden_dim = config.hidden_dim = config.hidden_size
        elif hasattr(config, "embed_dim"):
            self.hidden_dim = config.embed_dim
        else:
            self.hidden_dim = 1408 
            
        self.fc = nn.Linear(self.hidden_dim, 30)

    def forward(self, x):
        # [B, 3, 16, 224, 224] -> [B, 16, 3, 224, 224]
        x = x.permute(0, 2, 1, 3, 4) 
        outputs = self.backbone(x)
        
        features = None
        # 兼容性特征提取
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple) or isinstance(outputs, list):
            raw = outputs[0]
            if raw.dim() == 3: features = raw.mean(dim=1)
            elif raw.dim() == 2: features = raw
        elif isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3: features = outputs.mean(dim=1)
            elif outputs.dim() == 2: features = outputs
        
        logits = self.fc(features)
        return logits

# ================= 数据集 =================
class HRI30_Eval(Dataset):
    def __init__(self, root="/root/hri30/train"):
        self.data = []
        target_root = root if os.path.exists(root) and os.listdir(root) else "/root/hri30/train_set"
        print(f"Scanning data from: {target_root}")
        for i in range(1, 31):
            p = f"{target_root}/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'):
                    self.data.append((os.path.join(p, f), i-1))
        print(f"Found {len(self.data)} videos for evaluation.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        # 评估时使用均匀采样
        idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
        batch = vr.get_batch(idx)
        buffer = batch.asnumpy().transpose(0, 3, 1, 2)
        buffer = torch.from_numpy(buffer).float() / 255.0
        buffer = torch.nn.functional.interpolate(buffer, (IMG_SIZE, IMG_SIZE))
        return buffer, torch.tensor(label)

# ================= 执行评估 =================
if __name__ == "__main__":
    # 1. 准备数据
    ds = HRI30_Eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化模型并转为 BFloat16
    print("Initializing Model...")
    # 🔥 关键修复：直接在这里把整个模型转为 bfloat16，解决 dtype 冲突
    model = VideoMAEv2_Classifier(MODEL_ID).cuda().to(torch.bfloat16)

    # 3. 加载权重
    print(f"Loading State Dict...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(state_dict)
        print("✅ Weights Loaded Successfully!")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        exit()

    model.eval()
    
    # 4. 推理循环
    preds = []
    targets = []
    
    print("🔥 Running Inference...")
    with torch.no_grad():
        for x, y in tqdm(dl):
            # 输入也要转为 bfloat16
            x = x.cuda().to(torch.bfloat16)
            
            logits = model(x)
            
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            targets.extend(y.numpy())

    # 5. 计算指标
    acc = accuracy_score(targets, preds)
    print("\n" + "="*40)
    print(f"🏆 VideoMAE V2 Giant Baseline Result:")
    print(f"✅ Top-1 Accuracy: {acc*100:.2f}%")
    print("="*40)
