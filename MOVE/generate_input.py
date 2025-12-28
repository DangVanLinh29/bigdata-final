import os

# Đường dẫn chứa video (Cấu trúc thư mục local)
VIDEO_DIR = r"mapreduce_data\videos"
OUTPUT_FILE = r"mapreduce_data\input_list.txt"

# Lấy danh sách file
files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

if len(files) == 0:
    print("❌ Chưa có video nào trong mapreduce_data/videos!")
else:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for vid in files:
            f.write(vid + "\n")
    print(f"✅ Đã tạo file '{OUTPUT_FILE}' với {len(files)} video.")