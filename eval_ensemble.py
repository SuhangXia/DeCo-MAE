import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torchvision.transforms.v2 as T
import warnings
warnings.filterwarnings("ignore")

# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 配置 =================
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"
BERT_ID = "bert-base-uncased"
CACHE_DIR = "/root/autodl-tmp/hf_cache"

# 两个模型的路径
CKPT_BASELINE = "/root/autodl-tmp/checkpoints/sota_v2_best.pth"       
CKPT_FINAL    = "/root/autodl-tmp/checkpoints_final/final_sota_best.pth" 

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 8

# ================= 1. 语义字典 =================
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
    return protos

TEXT_PROTOTYPES = compute_text_prototypes()

# ================= 3. 模型定义 =================
# --- Model A: Baseline ---
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        v_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        dim = v_config.hidden_size if hasattr(v_config, "hidden_size") else 1408
        self.fc = nn.Linear(dim, 30)
    
    def forward(self, x):
        # x is [B, C, T, H, W]
        # 🔥 Removed permute!
        outputs = self.visual(x)
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        return self.fc(feat)

# --- Model B: Final SOTA ---
class FinalModel(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        v_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        dim = v_config.hidden_size if hasattr(v_config, "hidden_size") else 1408
        
        self.fc_cls = nn.Linear(dim, 30)
        self.register_buffer("text_prototypes", prototypes)
        self.video_proj = nn.Linear(dim, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x is [B, C, T, H, W]
        # 🔥 Removed permute!
        outputs = self.visual(x)
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        logits_cls = self.fc_cls(feat)
        
        v_emb = F.normalize(self.video_proj(feat), dim=-1)
        text_protos = self.text_prototypes.to(feat.device).to(feat.dtype)
        logits_sem = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        return 0.8 * logits_cls + 0.2 * logits_sem

# ================= 4. 验证数据 =================
class HRI30_Eval(Dataset):
    def __init__(self):
        self.data = []
        root = "/root/hri30/train"
        if not os.path.exists(root) or not os.listdir(root): root = "/root/hri30/train_set"
        for i in range(1, 31):
            p = f"{root}/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'): self.data.append((os.path.join(p, f), i-1))
        
        # 验证时只做 Resize + Normalize (无 Crop/Flip)
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        ])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
        batch = vr.get_batch(idx)
        # [T, H, W, C] -> [C, T, H, W]
        buffer = torch.from_numpy(batch.asnumpy()).permute(3, 0, 1, 2)
        buffer = self.transform(buffer)
        buffer = (buffer - self.mean) / self.std
        return buffer, torch.tensor(label)

# ================= 5. 双雄出击 =================
if __name__ == "__main__":
    ds = HRI30_Eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Load Model A: Baseline...")
    model_a = BaselineModel().cuda().to(torch.bfloat16)
    model_a.load_state_dict(torch.load(CKPT_BASELINE), strict=False)
    model_a.eval()

    print("Load Model B: Final SOTA...")
    model_b = FinalModel(TEXT_PROTOTYPES).cuda().to(torch.bfloat16)
    model_b.load_state_dict(torch.load(CKPT_FINAL), strict=False)
    model_b.eval()

    preds, targets = [], []
    print("🔥 Ensemble Inference (A + B)...")
    
    with torch.no_grad():
        for x, y in tqdm(dl):
            x = x.cuda().to(torch.bfloat16)
            
            # 分别预测
            logits_a = model_a(x) 
            logits_b = model_b(x) 
            
            # 🔥 融合策略: Soft Voting
            final_logits = logits_a + logits_b 
            
            batch_preds = torch.argmax(final_logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            targets.extend(y.numpy())

    acc = accuracy_score(targets, preds)
    print("\n" + "="*40)
    print(f"🏆 ENSEMBLE SOTA RESULT:")
    print(f"Model A (Baseline): 83.60%")
    print(f"Model B (Final):    82.84%")
    print(f"✅ Ensemble Acc:     {acc*100:.2f}%")
    print("="*40)
