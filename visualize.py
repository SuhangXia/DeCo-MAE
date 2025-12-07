import os
import torch
import torch.nn.functional as F
import decord
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import AutoModel, AutoConfig
import torchvision.transforms.v2 as T
import warnings
warnings.filterwarnings("ignore")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 配置 =================
import glob
video_files = glob.glob("/root/hri30/train/*/*.avi")
if len(video_files) > 0:
    idx = min(50, len(video_files)-1)
    VIDEO_PATH = video_files[idx]
else:
    VIDEO_PATH = "" 

CKPT_PATH = "/root/autodl-tmp/checkpoints_final/final_sota_best.pth"
MODEL_ID = "OpenGVLab/VideoMAEv2-giant"
CACHE_DIR = "/root/autodl-tmp/hf_cache"

NUM_FRAMES = 16
IMG_SIZE = 224

# ================= 模型定义 (智能 Hook) =================
class DualHeadMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        v_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
        v_config.use_cache = False
        self.visual = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, config=v_config, cache_dir=CACHE_DIR, torch_dtype=torch.float32)
        self.attention_map = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.attention_map = output.detach()

        target_module = None
        # 优先找 attn_drop
        for name, module in self.visual.named_modules():
            if "attn_drop" in name:
                target_module = module
        
        if target_module is not None:
            target_module.register_forward_hook(hook_fn)
            print("✅ Hooked Attention Layer")

    def forward(self, x):
        _ = self.visual(x)
        return self.attention_map

# ================= 图像处理 =================
def get_attention_map(model, video_tensor):
    model.eval()
    with torch.no_grad():
        _ = model(video_tensor)
    
    att_mat = model.attention_map
    if att_mat is None: return None

    # [B, Heads, N, N] -> Mean Heads -> [B, N, N]
    if att_mat.dim() == 4:
        att_mat = torch.mean(att_mat, dim=1)
    
    # 获取 [CLS] 的 attention
    # 假设第0个是CLS
    # 如果 N=2048 (无CLS?) 或者 N=2049 (有CLS)
    seq_len = att_mat.shape[-1]
    
    # 尝试取第0行
    cls_attn = att_mat[:, 0, :] # [B, N]
    
    # 如果包含自己，去掉自己
    # 这里我们做一个简单的处理：直接用全部
    # 归一化
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    return cls_attn

def visualize(video_path, save_path="attention_vis.png"):
    if not os.path.exists(video_path): return
    print(f"🎥 Video: {video_path}")
    
    # 读取
    vr = decord.VideoReader(video_path)
    idx = torch.linspace(0, len(vr)-1, NUM_FRAMES).long()
    batch = vr.get_batch(idx).asnumpy()
    
    # 预处理
    buffer = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])
    buffer = transform(buffer)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    norm_buffer = (buffer / 255.0 - mean) / std
    input_tensor = norm_buffer.permute(1, 0, 2, 3).unsqueeze(0).cuda()
    
    # 推理
    model = DualHeadMAE().cuda()
    try:
        sd = torch.load(CKPT_PATH)
        # 只加载 visual
        new_sd = {}
        for k, v in sd.items():
            if "visual" in k: new_sd[k.replace("visual.", "visual.")] = v 
            elif "backbone" in k: new_sd[k.replace("backbone.", "visual.")] = v
        model.load_state_dict(new_sd, strict=False)
        print("✅ Weights Loaded")
    except:
        print("⚠️ Random Weights")

    model.eval()
    attn_score = get_attention_map(model, input_tensor) # [1, N]
    
    # 🔥🔥🔥 暴力 Reshape 修复 🔥🔥🔥
    num_tokens = attn_score.shape[1]
    print(f"Tokens: {num_tokens}")
    
    # 目标：变成 [T, H, W]
    # 我们知道 T=8 (16/2)
    # 剩下的 spatial_tokens = num_tokens / 8
    
    # 假设有 CLS，先去掉一个看看能不能整除
    if num_tokens % 8 != 0:
        attn_score = attn_score[:, 1:] # 丢掉第一个
        num_tokens -= 1
    
    spatial = num_tokens // 8
    h = int(np.sqrt(spatial))
    w = h
    
    print(f"Reshaping to [8, {h}, {w}]")
    
    try:
        attn_score = attn_score.reshape(8, h, w)
    except:
        # 实在不行，硬插值
        print("⚠️ Shape mismatch, forcing interpolation...")
        attn_score = F.interpolate(attn_score.unsqueeze(0), size=8*14*14, mode='linear').reshape(8, 14, 14)

    # 插值回视频尺寸
    attn_score = F.interpolate(attn_score.unsqueeze(0).unsqueeze(0), size=(16, 224, 224), mode='trilinear').squeeze()
    attn_score = attn_score.cpu().numpy()
    
    # 绘图
    frame_indices = [2, 6, 10, 14]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    orig_imgs = F.interpolate(torch.from_numpy(batch).permute(0,3,1,2).float(), size=(224,224)).permute(0,2,3,1).numpy().astype(np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        img = orig_imgs[frame_idx]
        heatmap = attn_score[frame_idx]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Frame {frame_idx}")
        
        axes[1, i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Attention")
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    if VIDEO_PATH: visualize(VIDEO_PATH)
