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
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 1. 定义可见与不可见类 =================
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

UNSEEN_NAMES = [
    'MoveBackwardsWhilePolishing',
    'MoveRightWhileDrilling',
    'PickUpDrill',
    'WalkingWithObject',
    'NoCollaborativeWithPolisher'
]
UNSEEN_INDICES = [ALL_CLASSES.index(name) for name in UNSEEN_NAMES]
SEEN_INDICES = [i for i in range(len(ALL_CLASSES)) if i not in UNSEEN_INDICES]

# ================= 配置 =================
PRETRAINED_PATH = "/root/autodl-tmp/checkpoints/sota_v2_best.pth" 
BERT_ID = "bert-base-uncased"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"

BATCH_SIZE = 1 
GRAD_ACCUM = 32
EPOCHS = 10 # 增加到10轮

CACHE_DIR = "/root/autodl-tmp/hf_cache"
SAVE_DIR = "/root/autodl-tmp/checkpoints_zeroshot"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 2. 离线计算 Prototypes =================
def compute_text_prototypes():
    print("🚀 Computing Semantic Prototypes...")
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

# ================= 3. 数据集 =================
class HRI30_Split_Dataset(Dataset):
    def __init__(self, target_indices, root="/root/hri30/train"):
        self.data = []
        target_root = root if os.path.exists(root) and os.listdir(root) else "/root/hri30/train_set"
        target_set = set(target_indices)
        for i in range(1, 31):
            class_idx = i - 1
            if class_idx in target_set:
                p = f"{target_root}/{i}"
                if not os.path.exists(p): continue
                for f in os.listdir(p):
                    if f.endswith('.avi'):
                        self.data.append((os.path.join(p, f), class_idx))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        idx = torch.linspace(0, len(vr)-1, 16).long()
        batch = vr.get_batch(idx)
        buffer = batch.asnumpy().transpose(0, 3, 1, 2)
        buffer = torch.from_numpy(buffer).float() / 255.0
        buffer = torch.nn.functional.interpolate(buffer, (224, 224))
        return buffer, torch.tensor(label)

# ================= 4. 模型定义 =================
class DeCoMAE(nn.Module):
    def __init__(self, video_model_id, prototypes):
        super().__init__()
        v_config = AutoConfig.from_pretrained(video_model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(video_model_id, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        
        if hasattr(v_config, "hidden_size"): self.v_dim = v_config.hidden_size
        else: self.v_dim = 1408
        
        self.register_buffer("text_prototypes", prototypes) 
        self.video_proj = nn.Linear(self.v_dim, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

    def forward(self, x, labels=None):
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.visual(x)
        
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        v_emb = F.normalize(self.video_proj(feat), dim=-1)
        
        text_protos = self.text_prototypes.to(v_emb.device).to(v_emb.dtype)
        logits = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits

# ================= 5. 实验流程 =================
print("\n=== Phase 1: Training on SEEN Classes (25/30) ===")
train_ds = HRI30_Split_Dataset(SEEN_INDICES)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = DeCoMAE(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)

print("Loading Backbone...")
checkpoint = torch.load(PRETRAINED_PATH)
new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if "backbone" in k}
model.visual.load_state_dict(new_state_dict, strict=False)

# 🔥 分层学习率 + Scheduler 🔥
param_groups = [
    {'params': model.visual.parameters(), 'lr': 1e-5},
    {'params': model.video_proj.parameters(), 'lr': 1e-3},
    {'params': [model.logit_scale], 'lr': 1e-3}
]
opt = AdamW(param_groups, weight_decay=0.05)

# 动态调度器
num_training_steps = len(train_dl) * EPOCHS // GRAD_ACCUM
scheduler = get_cosine_schedule_with_warmup(
    opt, 
    num_warmup_steps=int(0.1 * num_training_steps), 
    num_training_steps=num_training_steps
)

model.train()
global_step = 0

for epoch in range(1, EPOCHS+1):
    pbar = tqdm(train_dl, desc=f"Train Epoch {epoch}/{EPOCHS}")
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
            scheduler.step() # 更新 LR
            opt.zero_grad()
            
        step_in_epoch += 1
        epoch_loss += loss.item() * GRAD_ACCUM
        
        # 显示当前 LR
        current_lr = scheduler.get_last_lr()[1] # 显示 projection 的大 LR
        pbar.set_postfix(loss=epoch_loss/step_in_epoch, lr=f"{current_lr:.1e}")

# Step B: 测试
print("\n=== Phase 2: Testing on UNSEEN Classes (Strict Mode) ===")
test_ds = HRI30_Split_Dataset(UNSEEN_INDICES)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

model.eval()
preds, targets = [], []

with torch.no_grad():
    for x, y in tqdm(test_dl):
        x = x.cuda().to(torch.bfloat16)
        _, logits = model(x)
        
        mask = torch.ones_like(logits) * float('-inf')
        mask[:, UNSEEN_INDICES] = 0
        masked_logits = logits + mask
        
        p = torch.argmax(masked_logits, dim=1).cpu().numpy()
        preds.extend(p)
        targets.extend(y.numpy())

acc = accuracy_score(targets, preds)
print("\n" + "="*50)
print(f"🚀 Strict Zero-Shot Results (with Dynamic LR)")
print(f"Training on: 25 classes")
print(f"Testing on:   5 classes (Unseen)")
print(f"✅ Accuracy: {acc*100:.2f}%")
print("="*50)
