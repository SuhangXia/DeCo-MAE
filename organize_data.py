import os
import csv
import shutil

# ================= 配置路径 (请根据你的实际情况检查) =================
# 假设你的原始视频文件都在这个文件夹里
SOURCE_VIDEOS_DIR = "/root/hri30/train_set"

# 目标文件夹：这是你最终训练脚本会读取的结构化目录
DEST_VIDEOS_DIR = "/root/hri30/train"

# 标签文件路径
LABELS_FILE = "/root/hri30/train_set_labels.csv"

# ================= 组织逻辑 =================
def organize_hri30_data():
    if not os.path.exists(LABELS_FILE):
        print(f"❌ 错误: 找不到标签文件 {LABELS_FILE}。请检查路径。")
        return

    if not os.path.exists(SOURCE_VIDEOS_DIR):
        print(f"❌ 错误: 找不到原始视频目录 {SOURCE_VIDEOS_DIR}。请检查路径。")
        return

    # 1. 创建目标根目录
    os.makedirs(DEST_VIDEOS_DIR, exist_ok=True)
    print(f"✅ 目标目录 {DEST_VIDEOS_DIR} 准备就绪。")

    # 2. 读取标签文件并组织
    success_count = 0
    fail_count = 0
    
    # 标签文件格式: [VideoID, ClassName, ClassID (1-30)]
    with open(LABELS_FILE, 'r') as f:
        reader = csv.reader(f)
        
        # ⚠️ 假设文件没有 header (表头)，直接从第一行开始读取
        for row in reader:
            if len(row) < 3:
                # 忽略空行或格式错误的行
                continue
            
            video_id_no_ext = row[0].strip()
            # Class ID 是第三列，我们需要用它来创建文件夹
            try:
                class_id = str(int(row[2].strip()))
            except ValueError:
                # 如果第三列不是数字，可能是 header 或者脏数据，跳过
                continue 

            # 3. 确定源文件路径
            found = False
            
            # 尝试 .avi 扩展名
            source_file_avi = os.path.join(SOURCE_VIDEOS_DIR, video_id_no_ext + ".avi")
            if os.path.exists(source_file_avi):
                source_path = source_file_avi
                found = True
            
            # 尝试 .mp4 扩展名
            source_file_mp4 = os.path.join(SOURCE_VIDEOS_DIR, video_id_no_ext + ".mp4")
            if not found and os.path.exists(source_file_mp4):
                source_path = source_file_mp4
                found = True
            
            if not found:
                print(f"⚠️ 找不到视频文件 {video_id_no_ext}.(avi/mp4)，跳过。")
                fail_count += 1
                continue

            # 4. 创建目标子目录
            dest_subdir = os.path.join(DEST_VIDEOS_DIR, class_id)
            os.makedirs(dest_subdir, exist_ok=True)

            # 5. 移动文件
            dest_path = os.path.join(dest_subdir, os.path.basename(source_path))
            shutil.move(source_path, dest_path)
            success_count += 1
            
            if success_count % 100 == 0:
                print(f"📦 已处理 {success_count} 个文件...")


    print("\n===================================")
    print(f"🎉 数据组织完成！")
    print(f"成功移动的文件总数: {success_count}")
    print(f"未找到的文件数: {fail_count}")
    print(f"现在，你的训练数据在 {DEST_VIDEOS_DIR} 目录中。")
    print("===================================")

if __name__ == "__main__":
    organize_hri30_data()
