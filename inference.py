import os
import csv
import torch
import torch.nn as nn  # 🔥 之前报错就是缺了这个
import torch.nn.functional as F
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
import torchvision.transforms.v2 as T
import warnings
warnings.filterwarnings("ignore")

# 🔥 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 配置 =================
# 使用最强的 Cool-down 模型 (85.80%)
CKPT_PATH = "/root/autodl-tmp/checkpoints_cooldown/cooldown_best.pth"
BERT_ID = "bert-base-uncased"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"
CACHE_DIR = "/root/autodl-tmp/hf_cache"

# 测试集路径
TEST_DIR = "/root/hri30/test_set" 
OUTPUT_FILE = "test_set_labels.csv"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 8 

# ================= 1. 语义字典 (用于 ID 转 Name) =================
# 必须保持顺序，对应 ID 0-29
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
# 列表索引 0 对应 label 0 (DeliverObject)
ID2LABEL = list(SEMANTIC_DICT.keys())

# ================= 2. 离线计算 Prototypes =================
def compute_text_prototypes():
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

TEXT_PROTOTYPES = compute_text_prototypes()

# ================= 3. 数据处理 =================
class VideoValTransform:
    def __init__(self):
        # 和 Cool-down 训练时保持一致：只 Resize，不 Crop
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

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        if not os.path.exists(root_dir):
            raise ValueError(f"测试集目录不存在: {root_dir}")
            
        files = sorted(os.listdir(root_dir)) # 排序保证顺序一致
        for f in files:
            if f.endswith('.avi') or f.endswith('.mp4'):
                full_path = os.path.join(root_dir, f)
                # 文件名无后缀: CIDxx_SIDxx_VIDxx
                vid_id = os.path.splitext(f)[0]
                self.data.append((full_path, vid_id))
        
        self.aug = VideoValTransform()
        print(f"📂 Found {len(self.data)} test videos in {root_dir}")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        path, vid_id = self.data[i]
        vr = decord.VideoReader(path)
        # 均匀采样
        idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
        batch = vr.get_batch(idx)
        # [T, H, W, C] -> [C, T, H, W]
        buffer = torch.from_numpy(batch.asnumpy()).permute(3, 0, 1, 2)
        buffer = self.aug(buffer)
        # -> [C, T, H, W]
        return buffer, vid_id

# ================= 4. Dual-Head 模型 =================
class DualHeadMAE(nn.Module):
    def __init__(self, video_model_id, prototypes):
        super().__init__()
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
        outputs = self.visual(x)
        if hasattr(outputs, 'last_hidden_state'): feat = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, tuple): feat = outputs[0].mean(dim=1) if outputs[0].dim()==3 else outputs[0]
        else: feat = outputs.mean(dim=1) if outputs.dim()==3 else outputs
        
        # Dual Head
        logits_cls = self.fc_cls(feat)
        
        v_emb = F.normalize(self.video_proj(feat), dim=-1)
        text_protos = self.text_prototypes.to(feat.device).to(feat.dtype)
        logits_sem = torch.matmul(v_emb, text_protos.t()) * self.logit_scale.exp()
        
        # 融合: 0.8 Cls + 0.2 Sem (保持和训练/验证时一致)
        return 0.8 * logits_cls + 0.2 * logits_sem

# ================= 5. 开始预测 =================
if __name__ == "__main__":
    # 1. 准备数据
    ds = TestDataset(TEST_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 加载模型
    print(f"Initializing Model (SOTA Cool-down)...")
    model = DualHeadMAE(MODEL_ID, TEXT_PROTOTYPES).cuda().to(torch.bfloat16)
    
    print(f"Loading Weights: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 推理
    results = []
    print("🔥 Generating predictions...")
    
    with torch.no_grad():
        for inputs, vid_ids in tqdm(dl):
            inputs = inputs.cuda().to(torch.bfloat16)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for vid_id, pred_idx in zip(vid_ids, preds):
                # pred_idx 是 0-29
                class_name = ID2LABEL[pred_idx]
                class_id = pred_idx + 1 # ⚠️ 关键：题目要求 ID 是 1-30，所以要 +1
                
                # 格式: VideoName, ClassName, ClassID
                results.append([vid_id, class_name, class_id])

    # 4. 写入 CSV
    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 根据 Training Set 格式，没有 Header，直接写数据
        writer.writerows(results)
    
    print(f"✅ 完成！文件已保存: {os.path.abspath(OUTPUT_FILE)}")
