import os, torch, decord
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

# --- 1. 定义数据集 (保持不变) ---
class HRI30(Dataset):
    def __init__(self):
        self.data = []
        for i in range(1,31):
            p = f"/root/hri30/train/{i}"
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.endswith('.avi'):
                    self.data.append((os.path.join(p,f), i-1))
        print(f"Found {len(self.data)} videos")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        path, label = self.data[i]
        vr = decord.VideoReader(path)
        idx = torch.linspace(0, len(vr)-1, 16).long()
        batch = vr.get_batch(idx)
        x = batch.asnumpy().transpose(0,3,1,2)
        x = torch.from_numpy(x).float() / 255.0
        x = torch.nn.functional.interpolate(x, (224,224))
        return x, torch.tensor(label)

ds = HRI30()
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True) # 保持 batch=16

# --- 2. 加载模型 (关键修改：加载 epoch 17) ---
print("正在加载 Epoch 17 的权重...")
# 初始化结构
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=30,
    ignore_mismatched_sizes=True
).cuda()

# 加载你之前训练好的权重
# 注意：确保目录下有 hri30_v2base_epoch17.pth
checkpoint = torch.load("hri30_v2base_epoch17.pth")
model.load_state_dict(checkpoint)
print("权重加载成功！准备继续训练...")

opt = torch.optim.AdamW(model.parameters(), 3e-5)

# --- 3. 训练循环 (从 18 开始，到 25) ---
step = 0
for epoch in range(18, 26): # 从 18 开始
    model.train()
    pbar = tqdm(dl, desc=f"Epoch {epoch}/25")
    for x,y in pbar:
        x,y = x.cuda(), y.cuda()
        loss = model(pixel_values=x, labels=y).loss
        loss.backward()
        step += 1
        
        # 梯度累积 (保持之前的逻辑)
        if step % 2 == 0:
            opt.step()
            opt.zero_grad()
            
        pbar.set_postfix(loss=loss.item())
    
    # --- 关键修改：只保存为 'latest.pth' 防止硬盘写满 ---
    # 如果你想保留最终的，可以在最后重命名
    save_name = "hri30_v2base_latest.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Epoch {epoch}/25 完成！模型已覆盖保存为 {save_name}")

print("训练全部完成！最终模型为 hri30_v2base_latest.pth")
