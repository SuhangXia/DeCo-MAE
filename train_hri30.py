step = 0
import os, torch, decord
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

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
        idx = torch.linspace(0, len(vr)-1, 16).long()  # 强制 16 帧
        batch = vr.get_batch(idx)                        # [16,H,W,3]
        x = batch.asnumpy().transpose(0,3,1,2)           # [16,3,H,W]
        x = torch.from_numpy(x).float() / 255.0
        x = torch.nn.functional.interpolate(x, (224,224))
        return x, torch.tensor(label)

ds = HRI30()
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

# 关键！用 V2 官方 base 权重（位置编码 1568，和 16×14×14 完全对齐）
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=30,
    ignore_mismatched_sizes=True
).cuda()

opt = torch.optim.AdamW(model.parameters(), 3e-5)

for epoch in range(1,26):
    model.train()
    pbar = tqdm(dl, desc=f"Epoch {epoch}/25")
    for x,y in pbar:
        x,y = x.cuda(), y.cuda()
        loss = model(pixel_values=x, labels=y).loss
        opt.zero_grad()
        loss.backward()
        step += 1
        if (step + 1) % 2 == 0:
            opt.step()
            opt.zero_grad()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), f"hri30_v2base_epoch{epoch}.pth")
    print(f"Epoch {epoch}/25 完成！模型已保存")
print("25 epoch 全部完成！模型在当前目录")
