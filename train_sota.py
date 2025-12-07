import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ================= 2025 SOTA 配置 (激进微调版) =================
MODEL_ID = "OpenGVLab/VideoMAEv2-giant" 

NUM_FRAMES = 16      
IMG_SIZE = 224
BATCH_SIZE = 1       
GRAD_ACCUM = 32      
# 🔥 修改 1: 学习率提升 20 倍
LR = 1e-4            
EPOCHS = 15          

CACHE_DIR = "/root/autodl-tmp/hf_cache"
SAVE_DIR = "/root/autodl-tmp/checkpoints"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🚀 SOTA 训练启动 | 显卡: RTX 5090 | 模型: V2 Giant | LR: {LR} (激进模式)")

# ================= 数据集 =================
class HRI30_SOTA(Dataset):
    def __init__(self, root="/root/hri30/train"):
        self.data = []
        target_root = root if os.path.exists(root) and os.listdir(root) else "/root/hri30/train_set"
        print(f"Loading data from: {target_root}")
        for i in range(1, 31):
            p = f"{target_root}/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'):
                    self.data.append((os.path.join(p, f), i-1))
        print(f"Found {len(self.data)} videos.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        # 增加一点随机性，防止过拟合
        if len(vr) > NUM_FRAMES:
            # 随机偏移采样
            start = np.random.randint(0, len(vr) - NUM_FRAMES)
            idx = torch.arange(start, start + NUM_FRAMES)
        else:
            idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
            
        batch = vr.get_batch(idx)
        buffer = batch.asnumpy().transpose(0, 3, 1, 2)
        buffer = torch.from_numpy(buffer).float() / 255.0
        buffer = torch.nn.functional.interpolate(buffer, (IMG_SIZE, IMG_SIZE))
        return buffer, torch.tensor(label)

# ================= 模型封装 =================
class VideoMAEv2_Classifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading Backbone: {model_id} ...")
        
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
            
        print(f"✅ Model Hidden Dimension: {self.hidden_dim}")
        self.fc = nn.Linear(self.hidden_dim, 30)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, labels=None):
        x = x.permute(0, 2, 1, 3, 4) 
        outputs = self.backbone(x)
        
        features = None
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple) or isinstance(outputs, list):
            raw = outputs[0]
            if raw.dim() == 3: features = raw.mean(dim=1)
            elif raw.dim() == 2: features = raw
        elif isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3: features = outputs.mean(dim=1)
            elif outputs.dim() == 2: features = outputs
        
        logits = self.fc(self.dropout(features))
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits

# ================= 训练流程 =================
ds = HRI30_SOTA()
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

model = VideoMAEv2_Classifier(MODEL_ID).cuda()

# 🔥 修改 2: 增加权重衰减
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# 🔥 修改 3: 加入 Cosine Scheduler (带 Warmup)
num_training_steps = len(dl) * EPOCHS // GRAD_ACCUM
num_warmup_steps = int(0.1 * num_training_steps) # 10% steps 用来热身
scheduler = get_cosine_schedule_with_warmup(
    opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

print("🔥 开始训练 V2 Giant (High LR + Scheduler)...")
model.train()
best_loss = 999.0

for epoch in range(1, EPOCHS+1):
    pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")
    epoch_loss = 0
    step = 0
    
    for x, y in pbar:
        x, y = x.cuda().to(torch.bfloat16), y.cuda()
        
        with autocast(dtype=torch.bfloat16):
            loss, logits = model(x, y)
            loss = loss / GRAD_ACCUM
            
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            # 梯度裁剪 (防止大 LR 导致梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step() # 更新学习率
            opt.zero_grad()
        
        step += 1
        current_loss = loss.item() * GRAD_ACCUM
        epoch_loss += current_loss
        
        # 显示当前 LR
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=current_loss, lr=f"{current_lr:.2e}")

    avg_loss = epoch_loss / len(dl)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/sota_v2_best.pth")
        print("🌟 Saved Best Model")

print("🏆 训练完成！")
