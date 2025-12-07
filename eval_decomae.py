import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc

# ================= 配置 =================
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"
BERT_ID = "bert-base-uncased"
CHECKPOINT_PATH = "/root/autodl-tmp/checkpoints_decomae/decomae_best.pth"
CACHE_DIR = "/root/autodl-tmp/hf_cache"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 8 

# 语义字典 (必须与训练一致)
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
ID2LABEL = list(SEMANTIC_DICT.keys())

# ================= 1. 离线计算语义原型 =================
def compute_prototypes():
    print("🚀 Pre-computing Semantic Prototypes...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID, cache_dir=CACHE_DIR)
    bert = AutoModel.from_pretrained(BERT_ID, cache_dir=CACHE_DIR).cuda()
    bert.eval()
    
    prompts = [f"A worker {SEMANTIC_DICT[c][0]} {SEMANTIC_DICT[c][1]} using {SEMANTIC_DICT[c][2]}" for c in ID2LABEL]
    
    with torch.no_grad():
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to('cuda')
        outputs = bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, dim=-1)
    
    protos = embeddings.cpu()
    del bert, tokenizer, inputs, outputs
    torch.cuda.empty_cache()
    return protos

TEXT_PROTOTYPES = compute_prototypes()

# ================= 2. DeCo-MAE 模型 =================
class DeCoMAE_Eval(nn.Module):
    def __init__(self, video_model_id, prototypes):
        super().__init__()
        print("Loading Video Backbone...")
        v_config = AutoConfig.from_pretrained(video_model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(video_model_id, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        
        if hasattr(v_config, "hidden_size"): self.v_dim = v_config.hidden_size
        elif hasattr(v_config, "embed_dim"): self.v_dim = v_config.embed_dim
        else: self.v_dim = 1408
        
        self.register_buffer("text_prototypes", prototypes)
        self.video_proj = nn.Linear(self.v_dim, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        v_out = self.visual(x)
        
        if hasattr(v_out, 'last_hidden_state'): v_feat = v_out.last_hidden_state.mean(dim=1)
        elif isinstance(v_out, tuple): v_feat = v_out[0].mean(dim=1) if v_out[0].dim()==3 else v_out[0]
        else: v_feat = v_out.mean(dim=1) if v_out.dim()==3 else v_out
        
        v_emb = self.video_proj(v_feat)
        v_emb = F.normalize(v_emb, dim=-1)
        
        text_protos = self.text_prototypes.to(v_emb.device).to(v_emb.dtype)
        logits = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        return logits

# ================= 3. 数据集 =================
class HRI30_Eval(Dataset):
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
        idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
        batch = vr.get_batch(idx)
        buffer = batch.asnumpy().transpose(0, 3, 1, 2)
        buffer = torch.from_numpy(buffer).float() / 255.0
        buffer = torch.nn.functional.interpolate(buffer, (IMG_SIZE, IMG_SIZE))
        return buffer, torch.tensor(label)

# ================= 4. 执行评估 =================
if __name__ == "__main__":
    ds = HRI30_Eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Initializing Model...")
    model = DeCoMAE_Eval(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)

    print(f"Loading Weights: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(state_dict) # 这里必须严格匹配
    model.eval()

    preds, targets = [], []
    print("🔥 Running Inference...")
    with torch.no_grad():
        for x, y in tqdm(dl):
            x = x.cuda().to(torch.bfloat16)
            logits = model(x)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(y.numpy())

    acc = accuracy_score(targets, preds)
    print("\n" + "="*40)
    print(f"🏆 DeCo-MAE (Semantic Decomposition) Result:")
    print(f"✅ Top-1 Accuracy: {acc*100:.2f}%")
    print("="*40)
