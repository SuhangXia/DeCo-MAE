import os
import gc
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
# 🔥🔥🔥 修正路径：指向 cooldown 文件夹 🔥🔥🔥
CHECKPOINT_PATH = "/root/autodl-tmp/checkpoints_cooldown/cooldown_best.pth"
BERT_ID = "bert-base-uncased"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 8  
CACHE_DIR = "/root/autodl-tmp/hf_cache"

print(f"🚀 开始终极评估 | 权重: {CHECKPOINT_PATH}")

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
    gc.collect()
    return protos

TEXT_PROTOTYPES = compute_text_prototypes()

# ================= 3. 验证集处理 =================
class VideoValTransform:
    def __init__(self):
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        ])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __call__(self, video_tensor):
        video_tensor = self.transform(video_tensor)
        video_tensor = (video_tensor - self.mean) / self.std
        return video_tensor

# ================= 4. 数据集 =================
class HRI30_Eval(Dataset):
    def __init__(self, root="/root/hri30/train"):
        self.data = []
        self.aug = VideoValTransform()
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
        idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
        batch = vr.get_batch(idx)
        # [T, H, W, C] -> [C, T, H, W]
        buffer = torch.from_numpy(batch.asnumpy()).permute(3, 0, 1, 2)
        buffer = self.aug(buffer)
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

    def forward(self, x):
        # x is [B, C, T, H, W]
        outputs = self.visual(x)
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        # Head 1 (Class)
        logits_cls = self.fc_cls(feat)
        
        # Head 2 (Semantic)
        v_emb = F.normalize(self.video_proj(feat), dim=-1)
        text_protos = self.text_prototypes.to(feat.device).to(feat.dtype)
        logits_sem = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        # 融合策略
        final_logits = 0.8 * logits_cls + 0.2 * logits_sem
        return final_logits

# ================= 6. 执行评估 =================
if __name__ == "__main__":
    ds = HRI30_Eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Initializing Model...")
    model = DualHeadMAE(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)

    print(f"Loading Weights: {CHECKPOINT_PATH}")
    # 注意：Cool-down 保存的是整个 state_dict，包括 backbone 和 heads
    state_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(state_dict)
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
    print(f"🏆 Final Cool-down Result:")
    print(f"✅ Top-1 Accuracy: {acc*100:.2f}%")
    print("="*40)
