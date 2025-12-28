import torch
import sys
import os
import cv2
import numpy as np

# Load Support Set
try:
    support = torch.load("support_set.pt", map_location="cpu")
    print(f"1. Support Shape: {support.shape}")
    
    # Kiểm tra xem c   dữ liệu không hay toàn số 0?
    if torch.sum(torch.abs(support)) == 0:
        print("🔴 NGUY HIỂM: Support Set toàn số 0! (Lỗi do OpenCV không đọc được video lúc tạo)")
    else:
        print(f"� Support Set OK. Sum value: {torch.sum(torch.abs(support)).item()}")
except Exception as e:
    print(f"🔴 Lỗi load Support Set: {e}")

# Test đọc 1 video bất kỳ
video_path = "/app/data/videos/city_video_0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        print(f"🟢 OpenCV đọc được video: {frame.shape}")
    else:
        print("🔴 OpenCV KHÔNG đọc được frame nào (Có thể thiếu thư viện codec)")
else:
    print("🔴 Không tìm thấy video test")

# Test nhanh Model (nếu có th)
try:
    sys.path.append(".")
    from libs.models.DMA.DMA_deploy import DMA_Deploy
    model = DMA_Deploy(n_way=10, backbone='resnet50')
    model.eval()
    print("🟢 Model initialized successfully")
except Exception as e:
    print(f"🔴 Lỗi init model: {e}")
