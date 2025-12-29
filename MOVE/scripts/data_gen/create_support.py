import torch
import cv2
import numpy as np
import os

# Cấu hình đường dẫn của bạn
VIDEO_DIR = r"C:\Nhom_3_BigData\MOVE\mapreduce_data\videos"
OUTPUT_FILE = "support_set.pt"
CLASS_NAMES = ["Run", "Jump", "Sit", "Walk", "Swim", "Cycling", "Diving", "Soccer", "Tennis", "Basketball"]

def read_and_process_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    # Padding nếu thiếu frame
    while len(frames) < 16 and len(frames) > 0:
        frames.append(frames[-1])
        
    if len(frames) == 0: return torch.zeros(16, 3, 224, 224)

    # Chuẩn hóa về [0,1] và đổi chiều: (Time, Height, Width, Channel) -> (Time, Channel, Height, Width)
    # Đây là định dạng CHUẨN mà DMA yêu cầu: (T, C, H, W)
    np_frames = np.array(frames, dtype=np.float32) / 255.0
    return torch.FloatTensor(np_frames).permute(0, 3, 1, 2)

print("Dang tao Support Set...")
support_tensors = []

# Duyệt qua từng folder class hoặc tìm file chứa tên class
# Ở đây giả định tất cả file nằm chung folder VIDEO_DIR
all_files = os.listdir(VIDEO_DIR)

for cls in CLASS_NAMES:
    found = False
    for fname in all_files:
        # Tìm video có tên chứa class (ví dụ: city_video_0 có thể đại diện cho Run?)
        # Nếu bạn không chắc video nào là class nào, script sẽ lấy đại diện theo thứ tự hoặc tên
        # Ở đây tôi lấy video đầu tiên tìm thấy (Demo), bạn nên chỉnh lại logic này nếu cần map chính xác
        if fname.endswith(".mp4"): 
            # Logic tạm: Lấy file bất kỳ nếu không tìm thấy tên class
            # Bạn có thể đổi thành: if cls.lower() in fname.lower():
            file_path = os.path.join(VIDEO_DIR, fname)
            tensor = read_and_process_video(file_path)
            support_tensors.append(tensor)
            print(f"Loaded class '{cls}' from {fname}")
            found = True
            break
    
    if not found:
        print(f"WARNING: Khong tim thay video cho class {cls}. Dung tensor rong.")
        support_tensors.append(torch.zeros(16, 3, 224, 224))

# Stack lại thành Tensor 5 chiều: (N_way, T, C, H, W)
# DMA yêu cầu: (Batch, N_way, K_shot, T, C, H, W) -> (1, 10, 1, 16, 3, 224, 224)
final_tensor = torch.stack(support_tensors) # (10, 16, 3, 224, 224)
final_tensor = final_tensor.unsqueeze(0).unsqueeze(2) # (1, 10, 1, 16, 3, 224, 224)

torch.save(final_tensor, OUTPUT_FILE)
print(f"DONE! Da luu file '{OUTPUT_FILE}' voi kich thuoc {final_tensor.shape}")