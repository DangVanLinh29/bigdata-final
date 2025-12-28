import os

# Đường dẫn thư mục chứa video cần đổi tên
folder_path = r"C:\Nhom_3_BigData\MOVE\mapreduce_data\temp" 
prefix = "city_video_" # Đổi thành "sport_video_" hoặc "daily_video_" tùy nhóm

files = os.listdir(folder_path)
for index, filename in enumerate(files):
    if filename.endswith(".mp4"):
        old_path = os.path.join(folder_path, filename)
        new_name = f"{prefix}{index}.mp4"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")