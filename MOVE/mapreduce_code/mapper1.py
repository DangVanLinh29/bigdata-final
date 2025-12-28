#!/usr/bin/env python3
import sys
import os
import cv2
import torch
import base64
import pickle
import numpy as np

# Mapper KHÔNG load model nữa để tránh lỗi.
# Nhiệm vụ: Đọc video -> Chuyển thành Tensor 5D -> Gửi cho Reducer.

def serialize_tensor(tensor):
    # Chuyển tensor sang bytes base64 để gửi qua Hadoop Streaming
    buffer = pickle.dumps(tensor.cpu().detach().numpy())
    return base64.b64encode(buffer).decode('utf-8')

def process_mapper(video_filename):
    video_path = os.path.join("/app/data/videos", video_filename.strip())
    
    if not os.path.exists(video_path): 
        return

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (224, 224))
        # Chuyển BGR sang RGB
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()

    if not frames: return
    while len(frames) < 16: frames.append(frames[-1])

    # 1. Chuẩn hóa về [0, 1] và Normalize theo chuẩn ImageNet
    np_frames = np.array(frames, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    np_frames = (np_frames - mean) / std
    
    # 2. Chuyển đổi kích thước (Shape)
    # Từ (16, 224, 224, 3) -> (16, 3, 224, 224)
    tensor = torch.FloatTensor(np_frames).permute(0, 3, 1, 2)
    
    # 3. Thêm chiều Batch để thành 5D: [1, 16, 3, 224, 224]
    # Đây chính là định dạng (B, T, C, H, W) mà DMA.py yêu cầu!
    tensor_5d = tensor.unsqueeze(0)

    try:
        serialized_data = serialize_tensor(tensor_5d)
        print(f"{video_filename.strip()}\t{serialized_data}")
    except Exception as e:
        sys.stderr.write(f"Error serializing {video_filename}: {str(e)}\n")

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if line:
            process_mapper(line)
