import os
import gc
# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 🔥 开启显存碎片整理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings("ignore")

# ================= 1. 语义分解词典 =================
SEMANTIC_DICT = {
    'DeliverObject':                       ('Deliver', 'Forward', 'Object'),
    'MoveBackwardsWhileDrilling':          ('Move', 'Backwards', 'Drill'),
    'MoveBackwardsWhilePolishing':         ('Move', 'Backwards', 'Polisher'),
    'MoveDiagonallyBackwardLeftWithDrill': ('Move', 'Diagonally Backward Left', 'Drill'),
    'MoveDiagonallyBackwardLeftWithPolisher': ('Move', 'Diagonally Backward Left', 'Polisher'),
    'MoveDiagonallyBackwardRightWithDrill': ('Move', 'Diagonally Backward Right', 'Drill'),
    'MoveDiagonallyBackwardRightWithPolisher': ('Move', 'Diagonally Backward Right', 'Polisher'),
    'MoveDiagonallyForwardLeftWithDrill':  ('Move', 'Diagonally Forward Left', 'Drill'),
    'MoveDiagonallyForwardLeftWithPolisher': ('Move', 'Diagonally Forward Left', 'Polisher'),
    'MoveDiagonallyForwardRightWithDrill': ('Move', 'Diagonally Forward Right', 'Drill'),
    'MoveDiagonallyForwardRightWithPolisher': ('Move', 'Diagonally Forward Right', 'Polisher'),
    'MoveForwardWhileDrilling':            ('Move', 'Forward', 'Drill'),
    'MoveForwardWhilePolishing':           ('Move', 'Forward', 'Polisher'),
    'MoveLeftWhileDrilling':               ('Move', 'Left', 'Drill'),
    'MoveLeftWhilePolishing':              ('Move', 'Left', 'Polisher'),
    'MoveRightWhileDrilling':              ('Move', 'Right', 'Drill'),
    'MoveRightWhilePolishing':             ('Move', 'Right', 'Polisher'),
    'NoCollaborativeWithDrilll':           ('Stand', 'No Action', 'Drill'),
    'NoCollaborativeWithPolisher':         ('Stand', 'No Action', 'Polisher'),
    'PickUpDrill':                         ('Pick Up', 'Upward', 'Drill'),
    'PickUpPolisher':                      ('Pick Up', 'Upward', 'Polisher'),
    'PickUpTheObject':                     ('Pick Up', 'Upward', 'Object'),
    'PutDownDrill':                        ('Put Down', 'Downward', 'Drill'),
    'PutDownPolisher':                     ('Put Down', 'Downward', 'Polisher'),
    'UsingTheDrill':                       ('Operate', 'Stationary', 'Drill'),
    'UsingThePolisher':                    ('Operate', 'Stationary', 'Polisher'),
    'Walking':                             ('Walk', 'Forward', 'Nothing'),
    'WalkingWithDrill':                    ('Walk', 'Forward', 'Drill'),
    'WalkingWithObject':                   ('Walk', 'Forward', 'Object'),
    'WalkingWithPolisher':                 ('Walk', 'Forward', 'Polisher')
}
ID2LABEL = list(SEMANTIC_DICT.keys())

# ================= 配置 =================
PRETRAINED_PATH = "/root/autodl-tmp/checkpoints/sota_v2_best.pth"
BERT_ID = "bert-base-uncased"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 1     # 必须保持 1，这是 5090 跑 Giant 的极限
GRAD_ACCUM = 32    
LR = 1e-5          
EPOCHS = 10        

CACHE_DIR = "/root/autodl-tmp/hf_cache"
SAVE_DIR = "/root/autodl-tmp/checkpoints_decomae"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 🚀 关键步骤：离线计算语义原型 =================
def compute_text_prototypes_offline():
    print("🚀 [Step 1] Loading BERT to compute Semantic Prototypes...")
    # 临时加载 BERT
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID, cache_dir=CACHE_DIR)
    bert = AutoModel.from_pretrained(BERT_ID, cache_dir=CACHE_DIR).cuda()
    bert.eval()
    
    prompts = []
    for class_name in ID2LABEL:
        action, direction, tool = SEMANTIC_DICT[class_name]
        prompt = f"A worker {action} {direction} using {tool}"
        prompts.append(prompt)
    
    print("   Computing embeddings...")
    with torch.no_grad():
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to('cuda')
        outputs = bert(**inputs)
        # [CLS] token pooling
        embeddings = outputs.last_hidden_state[:, 0, :] 
        embeddings = F.normalize(embeddings, dim=-1) # [30, 768]
    
    # 存到 CPU，准备卸磨杀驴
    prototypes = embeddings.cpu()
    
    # 🔥🔥🔥 核心操作：彻底删除 BERT，清空显存 🔥🔥🔥
    print("   Deleting BERT to free GPU memory...")
    del bert
    del tokenizer
    del inputs
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ BERT removed! Prototypes shape: {prototypes.shape}")
    return prototypes

# 先执行这一步，确保显存干净
TEXT_PROTOTYPES = compute_text_prototypes_offline()

# ================= 模型定义: DeCo-MAE (无 BERT 版) =================
class DeCoMAE(nn.Module):
    def __init__(self, video_model_id, prototypes):
        super().__init__()
        
        # 1. 视觉塔 (VideoMAE V2 Giant)
        print("🚀 [Step 2] Loading Video Backbone...")
        v_config = AutoConfig.from_pretrained(video_model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False # 省显存
        
        self.visual = AutoModel.from_pretrained(
            video_model_id, 
            trust_remote_code=True, 
            config=v_config, 
            cache_dir=CACHE_DIR, 
            torch_dtype=torch.bfloat16 # 必须用 bf16
        )
        
        # 自动获取维度
        if hasattr(v_config, "hidden_size"): self.v_dim = v_config.hidden_size
        elif hasattr(v_config, "embed_dim"): self.v_dim = v_config.embed_dim
        else: self.v_dim = 1408
        
        # 2. 注册语义原型 (作为 Buffer，不参与梯度更新)
        self.t_dim = 768
        self.register_buffer("text_prototypes", prototypes) 
        
        # 3. 跨模态投影层 (唯一新增的可训练参数)
        self.video_proj = nn.Linear(self.v_dim, self.t_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

    def forward(self, x, labels=None):
        # x: [B, 3, 16, 224, 224] -> [B, 16, 3, 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        
        # 1. 提取视频特征
        outputs = self.visual(x)
        
        # 鲁棒特征提取
        v_feat = None
        if hasattr(outputs, 'last_hidden_state'): v_feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): v_feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        elif isinstance(outputs, torch.Tensor): v_feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        # 2. 投影到语义空间
        v_emb = self.video_proj(v_feat) # [B, 1408] -> [B, 768]
        v_emb = F.normalize(v_emb, dim=-1)
        
        # 3. 计算相似度 (Video <-> 30个语义中心)
        # text_prototypes 已经在 GPU 上了 (register_buffer 自动处理)
        # 确保 dtype 一致 (bfloat16)
        text_protos = self.text_prototypes.to(v_emb.dtype)
        
        # Logits = Similarity * Temperature
        logits = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits

# ================= 数据集 =================
class HRI30_Dataset(Dataset):
    def __init__(self, root="/root/hri30/train"):
        self.data = []
        target_root = root if os.path.exists(root) and os.listdir(root) else "/root/hri30/train_set"
        print(f"Scanning {target_root}...")
        for i in range(1, 31):
            p = f"{target_root}/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'):
                    self.data.append((os.path.join(p, f), i-1))
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
        buffer = batch.asnumpy().transpose(0, 3, 1, 2)
        buffer = torch.from_numpy(buffer).float() / 255.0
        buffer = torch.nn.functional.interpolate(buffer, (IMG_SIZE, IMG_SIZE))
        return buffer, torch.tensor(label)

# ================= 训练流程 =================
ds = HRI30_Dataset()
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 🔥 初始化 DeCo-MAE (传入 BERT 计算好的原型)
model = DeCoMAE(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)

# 🔥 加载之前的 SOTA 权重 (Teacher)
print(f"🚀 [Step 3] Loading Pretrained Backbone from {PRETRAINED_PATH}...")
try:
    checkpoint = torch.load(PRETRAINED_PATH)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if "backbone" in k:
            new_key = k.replace("backbone.", "")
            new_state_dict[new_key] = v
    # 允许不匹配 (因为多了 video_proj，少了 fc)
    model.visual.load_state_dict(new_state_dict, strict=False)
    print("✅ Backbone Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load backbone ({e}). Training from scratch (NOT RECOMMENDED).")

opt = AdamW(model.parameters(), lr=LR, weight_decay=0.05)

print("🔥 开始 DeCo-MAE 训练 (Robust Mode)...")
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
            opt.step()
            opt.zero_grad()
        
        step += 1
        current_loss = loss.item() * GRAD_ACCUM
        epoch_loss += current_loss
        pbar.set_postfix(loss=current_loss)

    avg_loss = epoch_loss / len(dl)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/decomae_best.pth")
        print("🌟 Saved Best DeCo-MAE Model")

print("🏆 语义对齐训练完成！")
