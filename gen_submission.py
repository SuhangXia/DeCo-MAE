import os
import torch
import decord
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

# === 1. 配置与标签映射 ===
# 完整的 30 类映射表
ID2LABEL = {
    1: 'DeliverObject', 2: 'MoveBackwardsWhileDrilling', 3: 'MoveBackwardsWhilePolishing', 
    4: 'MoveDiagonallyBackwardLeftWithDrill', 5: 'MoveDiagonallyBackwardLeftWithPolisher', 
    6: 'MoveDiagonallyBackwardRightWithDrill', 7: 'MoveDiagonallyBackwardRightWithPolisher', 
    8: 'MoveDiagonallyForwardLeftWithDrill', 9: 'MoveDiagonallyForwardLeftWithPolisher', 
    10: 'MoveDiagonallyForwardRightWithDrill', 11: 'MoveDiagonallyForwardRightWithPolisher', 
    12: 'MoveForwardWhileDrilling', 13: 'MoveForwardWhilePolishing', 14: 'MoveLeftWhileDrilling', 
    15: 'MoveLeftWhilePolishing', 16: 'MoveRightWhileDrilling', 17: 'MoveRightWhilePolishing', 
    18: 'NoCollaborativeWithDrilll', 19: 'NoCollaborativeWithPolisher', 20: 'PickUpDrill', 
    21: 'PickUpPolisher', 22: 'PickUpTheObject', 23: 'PutDownDrill', 24: 'PutDownPolisher', 
    25: 'UsingTheDrill', 26: 'UsingThePolisher', 27: 'Walking', 28: 'WalkingWithObject', 
    29: 'WalkingWithDrill', 30: 'WalkingWithPolisher'
}

# 路径配置
TEST_DIR = "/root/hri30/test_set"
OUTPUT_FILE = "submission.csv"

# 优先寻找最好的模型
if os.path.exists("baseline_v1_100acc.pth"):
    MODEL_PATH = "baseline_v1_100acc.pth"
elif os.path.exists("/root/baseline_v1_100acc.pth"):
    MODEL_PATH = "/root/baseline_v1_100acc.pth"
else:
    # 如果找不到好模型，再尝试找 latest
    MODEL_PATH = "hri30_v2base_latest.pth"

print(f"🎯 选定的模型文件: {MODEL_PATH}")

# === 2. 定义测试集 Dataset ===
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        if not os.path.exists(root_dir):
            raise ValueError(f"测试集目录不存在: {root_dir}")
            
        files = os.listdir(root_dir)
        for f in files:
            if f.endswith('.avi') or f.endswith('.mp4'):
                full_path = os.path.join(root_dir, f)
                # 假设文件名是 "CID01_SID01_VID01.avi" -> ID 是 "CID01_SID01_VID01"
                vid_id = os.path.splitext(f)[0]
                self.data.append((full_path, vid_id))
        
        # 按文件名排序，保证输出顺序整齐
        self.data.sort(key=lambda x: x[1])
        print(f"📂 找到 {len(self.data)} 个测试视频")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        path, vid_id = self.data[i]
        vr = decord.VideoReader(path)
        # 这里的 16 必须和训练时保持一致
        idx = torch.linspace(0, len(vr)-1, 16).long()
        batch = vr.get_batch(idx)
        x = batch.asnumpy().transpose(0,3,1,2)
        x = torch.from_numpy(x).float() / 255.0
        x = torch.nn.functional.interpolate(x, (224,224))
        return x, vid_id

# === 3. 开始预测 ===
def generate_csv():
    # 加载模型
    print("正在加载模型结构...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=30,
        ignore_mismatched_sizes=True
    ).cuda()
    
    print(f"正在加载权重: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ 模型加载失败! 文件可能损坏: {e}")
        return

    model.eval()

    # 数据集
    try:
        ds = TestDataset(TEST_DIR)
    except ValueError as e:
        print(e)
        return

    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    results = []
    
    print("🚀 开始推理 (Inference)...")
    with torch.no_grad():
        for inputs, vid_ids in tqdm(dl):
            inputs = inputs.cuda()
            outputs = model(inputs).logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for vid_id, pred_idx in zip(vid_ids, preds):
                # ID2LABEL 是 1-30，pred_idx 是 0-29，所以要 +1
                final_label_id = pred_idx + 1 
                
                if final_label_id in ID2LABEL:
                    label_name = ID2LABEL[final_label_id]
                else:
                    label_name = "Unknown"
                
                # 写入三列: ID, Name, ClassID
                results.append([vid_id, label_name, final_label_id])

    # === 4. 写入 CSV ===
    print(f"正在写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    print(f"✅ 大功告成！提交文件已生成: {os.path.abspath(OUTPUT_FILE)}")
    print("请将该文件下载并发送给老师。")

if __name__ == "__main__":
    generate_csv()
