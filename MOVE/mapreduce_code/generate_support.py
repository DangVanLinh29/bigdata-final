import torch
import cv2
import numpy as np
import os

# Cấu hình đường dẫn nội bộ Docker
VIDEO_DIR = "/app/data/videos"
OUTPUT_FILE = "/app/code/support_set.pt"
CLASS_NAMES = ["Run", "Jump", "Sit", "Walk", "Swim", "Cycling", "Diving", "Soccer", "Tennis", "Basketball"]

def read_and_process_video(path):
    if not os.path.exists(path): return None
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    while len(frames) < 16 and len(frames) > 0:
        frames.append(frames[-1])
        
    if len(frames) == 0: return torch.zeros(16, 3, 224, 224)

    # Chuẩn hóa về [0,1] và đổi chiều: (T, H, W, C) -> (T, C, H, W)
    np_frames = np.array(frames, dtype=np.float32) / 255.0
    return torch.FloatTensor(np_frames).permute(0, 3, 1, 2)

print("Dang tao Support Set tu:", VIDEO_DIR)
support_tensors = []
all_files = os.listdir(VIDEO_DIR)

for cls in CLASS_NAMES:
    found = False
    for fname in all_files:
        # Logic tìm file đơn giản (có thể cải thiện nếu cần map chính xác)
        if fname.endswith(".mp4"): 
            # Tìm video đại diện (lấy đại video đầu tiên)
            # Vì trong Docker bạn thường chỉ có 1 vài video test, lấy đại cũng được để chạy luồng
            file_path = os.path.join(VIDEO_DIR, fname)
            tensor = read_and_process_video(file_path)
            if tensor is not None:
                support_tensors.append(tensor)
                print(f"Loaded class '{cls}' from {fname}")
                found = True
                break
    
    if not found:
        print(f"WARNING: Khong tim thay video cho class {cls}. Dung tensor rong.")
        support_tensors.append(torch.zeros(16, 3, 224, 224))

# Stack thành Tensor 5 chiều: (1, 10, 1, 16, 3, 224, 224)
# Lưu ý shape này khớp với code mapper mới: (B, N, K, T, C, H, W)
final_tensor = torch.stack(support_tensors) # (10, 16, 3, 224, 224)
final_tensor = final_tensor.unsqueeze(0).unsqueeze(2) # (1, 10, 1, 16, 3, 224, 224)

torch.save(final_tensor, OUTPUT_FILE)
print(f"DONE! Da luu file '{OUTPUT_FILE}' voi kich thuoc {final_tensor.shape}")
