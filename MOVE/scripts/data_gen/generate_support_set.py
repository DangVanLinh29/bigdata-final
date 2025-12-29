import torch
import cv2
import numpy as np
import os
import sys

# --- CẤU HÌNH KHỚP 100% VỚI ẢNH EM GỬI ---
# Thầy xếp lại theo ABC cho khớp với ảnh của em để dễ kiểm tra
CLASS_NAMES = [
    "Archery", "Boxing", "Cycling", "Diving", "Driving", "Flying", 
    "Jump", "Rowing", "Run", "Skateboarding", "Skiing", "Walk"
]

VIDEO_FOLDER = "ref_videos"  # Tên thư mục chứa video
OUTPUT_FILE = "support_set.pt"
NUM_FRAMES = 16
IMG_SIZE = 224

def load_video_to_tensor(video_path):
    # Kiểm tra file tồn tại
    if not os.path.exists(video_path):
        print(f"❌ LỖI TO: Không tìm thấy file '{video_path}'")
        return None # Trả về None để lọc sau

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize & Normalize
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        frames.append(frame)
    cap.release()
    
    # Pad nếu thiếu frame
    if not frames: 
        print(f"⚠️ Cảnh báo: Video {video_path} bị lỗi (không đọc được frame nào).")
        return torch.zeros(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)
        
    while len(frames) < NUM_FRAMES: 
        frames.append(frames[-1])
    
    # Lấy đúng 16 frame
    frames = frames[:NUM_FRAMES]
    
    # Chuyển đổi dimension: [T, H, W, C] -> [T, C, H, W] -> [1, T, C, H, W]
    tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor

def main():
    print(f"⏳ Đang xử lý {len(CLASS_NAMES)} video từ thư mục '{VIDEO_FOLDER}'...")
    support_tensors = []
    
    for action in CLASS_NAMES:
        # Code tự động ghép tên: Archery -> ref_videos/Archery.mp4
        video_path = os.path.join(VIDEO_FOLDER, f"{action}.mp4")
        
        print(f"   + Đang đọc class '{action}' từ file: {video_path}")
        
        tensor = load_video_to_tensor(video_path)
        
        if tensor is None:
            print("🛑 Dừng chương trình vì thiếu file. Kiểm tra lại tên file đi em!")
            sys.exit(1)
            
        support_tensors.append(tensor)

    # Gộp lại thành 1 Tensor lớn
    # Shape: [12, 1, 16, 3, 224, 224]
    final_tensor = torch.stack(support_tensors, dim=0)
    
    # Thêm chiều Batch: [1, 12, 1, 16, 3, 224, 224]
    final_tensor = final_tensor.unsqueeze(0)
    
    torch.save(final_tensor, OUTPUT_FILE)
    print("\n" + "="*50)
    print(f"🎉 THÀNH CÔNG! Đã tạo file '{OUTPUT_FILE}'")
    print(f"📦 Kích thước chuẩn: {final_tensor.shape}")
    print("="*50)

if __name__ == "__main__":
    if not os.path.exists(VIDEO_FOLDER):
        print(f"❌ Không thấy thư mục '{VIDEO_FOLDER}'. Em đang đứng ở thư mục nào thế?")
        print(f"📂 Thư mục hiện tại: {os.getcwd()}")
    else:
        main()