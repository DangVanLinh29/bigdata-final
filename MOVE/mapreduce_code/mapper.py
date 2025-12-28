#!/usr/bin/env python3
import sys
import os
sys.path.append(".") 
import torch
import cv2
import numpy as np
import json

# Import class sạch từ Bước 1
from libs.models.DMA.DMA_deploy import DMA_Deploy 

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Run", "Jump", "Sit", "Walk", "Swim", "Cycling", "Diving", "Soccer", "Tennis", "Basketball"]

# 1. LOAD MODEL
# Lưu ý: Class này đã tự xử lý pretrained=False và fix kích thước 29->28
model = DMA_Deploy(n_way=len(CLASS_NAMES), backbone='resnet50')
model.to(DEVICE)
model.eval()

# Load weights
if os.path.exists("latest_checkpoint.pth"):
    try:
        ckpt = torch.load("latest_checkpoint.pth", map_location=DEVICE, weights_only=False)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        sys.stderr.write("Weights loaded successfully.\n")
    except Exception as e:
        sys.stderr.write(f"Weight load error: {e}\n")

# 2. LOAD SUPPORT SET (TỪ FILE .PT CỦA BƯỚC 2)
if os.path.exists("support_set.pt"):
    SUPPORT_TENSOR = torch.load("support_set.pt", map_location=DEVICE, weights_only=False)
    sys.stderr.write(f"Support set loaded: {SUPPORT_TENSOR.shape}\n")
else:
    sys.stderr.write("CRITICAL: support_set.pt not found! Using fallback zeros.\n")
    SUPPORT_TENSOR = torch.zeros(1, 10, 1, 16, 3, 224, 224).to(DEVICE)

SUPPORT_MASK = torch.ones((1, 10, 1, 16, 224, 224)).to(DEVICE)

# 3. HÀM XỬ LÝ
def process_video(video_filename):
    video_path = os.path.join("/app/data/videos", video_filename.strip())
    if not os.path.exists(video_path): return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (224, 224))
        frames.append(f)
    cap.release()
    
    if not frames: return None
    while len(frames) < 16: frames.append(frames[-1])
    
    # Chuẩn hóa Input Query khớp với Support Set
    # (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
    # Shape cuối: (1, 16, 3, 224, 224)
    np_frames = np.array(frames, dtype=np.float32) / 255.0
    tens = torch.FloatTensor(np_frames).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Gọi model (DMA_Deploy đã fix lỗi forward bên trong)
        # Model trả về tuple, lấy phần tử đầu tiên là mask/score
        res = model(tens, SUPPORT_TENSOR, SUPPORT_MASK)
        
        # Xử lý output tùy vào cấu trúc trả về của DMA
        # Giả sử trả về tuple (mask, proposal, cls)
        if isinstance(res, tuple): 
            outputs = res[2] # Thường cls nằm ở cuối hoặc index 2
        else:
            outputs = res
            
        # Flatten để lấy score
        scores = outputs
        if isinstance(scores, tuple): scores = scores[0] # Phòng hờ
        while scores.dim() > 2: scores = scores.mean(dim=-1)
        
        # Softmax & Label
        probs = torch.nn.functional.softmax(scores, dim=1)
        conf, idx = torch.max(probs, 1)
        label = CLASS_NAMES[idx.item()] if idx.item() < len(CLASS_NAMES) else "Unknown"
        
        return {
            "label": label,
            "conf": round(conf.item(), 4),
            "filename": video_filename,
            "logits": scores.flatten().tolist()
        }

# --- MAIN LOOP ---
import time
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        sys.stderr.write(f"Dang xu ly video: {line} ...\n") 
        start_t = time.time()
        res = process_video(line)
        if res:
            end_t = time.time()
            sys.stderr.write(f"-> Xong {line} trong {end_t - start_t:.2f}s\n")
            print(f"{res['label']}\t{json.dumps(res)}")
    except Exception as e:
        sys.stderr.write(f"Error {line}: {e}\n")