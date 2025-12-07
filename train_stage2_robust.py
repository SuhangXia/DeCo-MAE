import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast
import torchvision.transforms.v2 as T
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 1. 语义字典 (30类) =================
SEMANTIC_DICT = {
    'DeliverObject': ('Deliver', 'Forward', 'Object'),
    'MoveBackwardsWhileDrilling': ('Move', 'Backwards', 'Drill'),
    'MoveBackwardsWhilePolishing': ('Move', 'Backwards', 'Polisher'),
    'MoveDiagonallyBackwardLeftWithDrill': ('Move', 'Diagonally Backward Left', 'Drill'),
    'MoveDiagonallyBackwardLeftWithPolisher': ('Move', 'Diagonally Backward Left', 'Polisher'),
    'MoveDiagonallyBackwardRightWithDrill': ('Move', 'Diagonally Backward Right', 'Drill'),
    'MoveDiagonallyBackwardRightWithPolisher': ('Move', 'Diagonally Backward Right', 'Polisher'),
    'MoveDiagonallyForwardLeftWithDrill': ('Move', 'Diagonally Forward Left', 'Drill'),
    'MoveDiagonallyForwardLeftWithPolisher': ('Move', 'Diagonally Forward Left', 'Polisher'),
    'MoveDiagonallyForwardRightWithDrill': ('Move', 'Diagonally Forward Right', 'Drill'),
    'MoveDiagonallyForwardRightWithPolisher': ('Move', 'Diagonally Forward Right', 'Polisher'),
    'MoveForwardWhileDrilling': ('Move', 'Forward', 'Drill'),
    'MoveForwardWhilePolishing': ('Move', 'Forward', 'Polisher'),
    'MoveLeftWhileDrilling': ('Move', 'Left', 'Drill'),
    'MoveLeftWhilePolishing': ('Move', 'Left', 'Polisher'),
    'MoveRightWhileDrilling': ('Move', 'Right', 'Drill'),
    'MoveRightWhilePolishing': ('Move', 'Right', 'Polisher'),
    'NoCollaborativeWithDrilll': ('Stand', 'No Action', 'Drill'),
    'NoCollaborativeWithPolisher': ('Stand', 'No Action', 'Polisher'),
    'PickUpDrill': ('Pick Up', 'Upward', 'Drill'),
    'PickUpPolisher': ('Pick Up', 'Upward', 'Polisher'),
    'PickUpTheObject': ('Pick Up', 'Upward', 'Object'),
    'PutDownDrill': ('Put Down', 'Downward', 'Drill'),
    'PutDownPolisher': ('Put Down', 'Downward', 'Polisher'),
    'UsingTheDrill': ('Operate', 'Stationary', 'Drill'),
    'UsingThePolisher': ('Operate', 'Stationary', 'Polisher'),
    'Walking': ('Walk', 'Forward', 'Nothing'),
    'WalkingWithDrill': ('Walk', 'Forward', 'Drill'),
    'WalkingWithObject': ('Walk', 'Forward', 'Object'),
    'WalkingWithPolisher': ('Walk', 'Forward', 'Polisher')
}
ALL_CLASSES = list(SEMANTIC_DICT.keys())

# ================= 配置 =================
PRETRAINED_PATH = "/root/autodl-tmp/checkpoints/sota_v2_best.pth" 
BERT_ID = "bert-base-uncased"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 1     
GRAD_ACCUM = 32    
LR = 2e-5          
EPOCHS = 30        

CACHE_DIR = "/root/autodl-tmp/hf_cache"
SAVE_DIR = "/root/autodl-tmp/checkpoints_final"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 2. 离线计算 Prototypes =================
def compute_text_prototypes():
    print("🚀 Pre-computing Semantic Prototypes...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID, cache_dir=CACHE_DIR)
    bert = AutoModel.from_pretrained(BERT_ID, cache_dir=CACHE_DIR).cuda()
    bert.eval()
    prompts = [f"A worker {SEMANTIC_DICT[c][0]} {SEMANTIC_DICT[c][1]} using {SEMANTIC_DICT[c][2]}" for c in ALL_CLASSES]
    with torch.no_grad():
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to('cuda')
        outputs = bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :] 
        embeddings = F.normalize(embeddings, dim=-1)
    protos = embeddings.cpu()
    del bert, tokenizer, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return protos

TEXT_PROTOTYPES = compute_text_prototypes()

# ================= 3. 数据增强 (手动 Normalize 版) =================
class VideoAugmentation:
    def __init__(self):
        # ⚠️ 删除了 T.Normalize，改为手动实现
        self.spatial_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
        ])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __call__(self, video_tensor):
        # input: [C, T, H, W]
        # 1. 空间增强
        video_tensor = self.spatial_transform(video_tensor)
        
        # 2. 手动 Normalize (广播机制，绝对不会错)
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor

# ================= 4. 完整数据集 =================
class HRI30_FullDataset(Dataset):
    def __init__(self, root="/root/hri30/train"):
        self.data = []
        self.aug = VideoAugmentation() 
        target_root = root if os.path.exists(root) and os.listdir(root) else "/root/hri30/train_set"
        print(f"Scanning FULL dataset from {target_root}...")
        for i in range(1, 31):
            p = f"{target_root}/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'):
                    self.data.append((os.path.join(p, f), i-1))
        print(f"Loaded {len(self.data)} videos.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        if len(vr) > NUM_FRAMES:
            start = np.random.randint(0, len(vr) - NUM_FRAMES)
            idx = torch.arange(start, start + NUM_FRAMES)
        else:
            idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
            
        batch = vr.get_batch(idx)
        # [T, H, W, C] -> [C, T, H, W]
        buffer = torch.from_numpy(batch.asnumpy()).permute(3, 0, 1, 2) 
        
        # 应用增强 (包含手动 Norm)
        buffer = self.aug(buffer)
        
        # 还原为 [C, T, H, W] -> DataLoader -> [B, C, T, H, W]
        # VideoMAE 需要 [B, C, T, H, W]
        return buffer, torch.tensor(label)

# ================= 5. Dual-Head 模型 =================
class DualHeadMAE(nn.Module):
    def __init__(self, video_model_id, prototypes):
        super().__init__()
        print("Loading Video Backbone...")
        v_config = AutoConfig.from_pretrained(video_model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(video_model_id, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        
        if hasattr(v_config, "hidden_size"): self.v_dim = v_config.hidden_size
        else: self.v_dim = 1408
        
        self.fc_cls = nn.Linear(self.v_dim, 30)
        
        self.register_buffer("text_prototypes", prototypes)
        self.video_proj = nn.Linear(self.v_dim, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, labels=None):
        # x is [B, C, T, H, W]
        outputs = self.visual(x)
        
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        feat = self.dropout(feat)
        
        logits_cls = self.fc_cls(feat)
        
        v_emb = F.normalize(self.video_proj(feat), dim=-1)
        text_protos = self.text_prototypes.to(feat.device).to(feat.dtype)
        logits_sem = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        loss = None
        if labels is not None:
            loss_cls = F.cross_entropy(logits_cls, labels)
            loss_sem = F.cross_entropy(logits_sem, labels)
            loss = 0.7 * loss_cls + 0.3 * loss_sem
            
        return loss, logits_cls

# ================= 6. 训练流程 =================
print("\n=== STARTING FINAL SOTA TRAINING (30 EPOCHS) ===")
ds = HRI30_FullDataset()
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

model = DualHeadMAE(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)

print(f"Loading weights from {PRETRAINED_PATH}...")
checkpoint = torch.load(PRETRAINED_PATH)
new_sd = {}
for k, v in checkpoint.items():
    if "backbone" in k: new_sd[k.replace("backbone.", "")] = v
    elif "fc" in k: new_sd["fc_cls." + k.replace("fc.", "")] = v
model.visual.load_state_dict(new_sd, strict=False)
model.load_state_dict(new_sd, strict=False)

param_groups = [
    {'params': model.visual.parameters(), 'lr': 1e-5},
    {'params': model.fc_cls.parameters(), 'lr': 1e-4},
    {'params': model.video_proj.parameters(), 'lr': 1e-3},
    {'params': [model.logit_scale], 'lr': 1e-3}
]
opt = AdamW(param_groups, weight_decay=0.05)

total_steps = len(dl) * EPOCHS // GRAD_ACCUM
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

model.train()
global_step = 0
best_loss = 999.0

for epoch in range(1, EPOCHS+1):
    pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")
    epoch_loss = 0
    step_in_epoch = 0
    
    for x, y in pbar:
        x, y = x.cuda().to(torch.bfloat16), y.cuda()
        with autocast(dtype=torch.bfloat16):
            loss, _ = model(x, y)
            loss = loss / GRAD_ACCUM
        loss.backward()
        
        global_step += 1
        if global_step % GRAD_ACCUM == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()
            
        step_in_epoch += 1
        epoch_loss += loss.item() * GRAD_ACCUM
        pbar.set_postfix(loss=epoch_loss/step_in_epoch, lr=f"{scheduler.get_last_lr()[0]:.1e}")

    avg_loss = epoch_loss / len(dl)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/final_sota_best.pth")
        print("🌟 Saved Best Final Model")
        
    torch.save(model.state_dict(), f"{SAVE_DIR}/final_sota_latest.pth")

print("🏆 最终 SOTA 训练完成！")
