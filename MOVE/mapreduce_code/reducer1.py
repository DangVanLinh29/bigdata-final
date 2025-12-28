#!/usr/bin/env python3
import sys
import os
import torch
import base64
import pickle
import numpy as np
import json
from collections import OrderedDict

# Setup đường dẫn
sys.path.append(".")
try:
    from libs.models.DMA.DMA_deploy import DMA_Deploy
except ImportError:
    if os.path.exists("libs"):
        sys.path.append("libs")
        from libs.models.DMA.DMA_deploy import DMA_Deploy
    else:
        raise

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Run", "Jump", "Driving", "Flying", "Walk", "Cycling", "Diving", "Boxing", "Skiing", "Rowing", "Skateboarding", "Archery"]

# 1. Khởi to Model
model = DMA_Deploy(n_way=len(CLASS_NAMES), backbone='resnet50')
model.to(DEVICE)
model.eval()

# 2. LOAD WEIGHTS THÔNG MINH (FIX LỖI RANDOM)
if os.path.exists("latest_checkpoint.pth"):
    try:
        # Load checkpoint
        try:
            ckpt = torch.load("latest_checkpoint.pth", map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load("latest_checkpoint.pth", map_location=DEVICE)
            
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # --- FIX LỖI TIỀN TỐ 'module.' ---
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k # Xóa 'module.' nếu có
            new_state_dict[name] = v
            
        # Load lại với strict=False nhưng hy vọng khớp nhiều hơn
        model.load_state_dict(new_state_dict, strict=False)
        # Ghi log nhẹ để biết đã load
        sys.stderr.write("[INFO] Weights loaded successfully with module fix.\n")
        
    except Exception as e:
        sys.stderr.write(f"[ERROR] Loading weights failed: {e}\n")

# 3. Load Support Set (Kiểm tra kỹ)
if os.path.exists("support_set.pt"):
    try:
        try:
            SUPPORT_TENSOR = torch.load("support_set.pt", map_location=DEVICE, weights_only=False)
        except:
            SUPPORT_TENSOR = torch.load("support_set.pt", map_location=DEVICE)
        sys.stderr.write(f"[INFO] Support Set Loaded. Shape: {SUPPORT_TENSOR.shape}\n")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Loading Support Set failed: {e}\n")
        # Thay 10 bằng len(CLASS_NAMES)
        SUPPORT_TENSOR = torch.zeros(1, len(CLASS_NAMES), 1, 16, 3, 224, 224).to(DEVICE)
else:
    sys.stderr.write("[WARNING] No support_set.pt found. Using Zeros.\n")
    # Thay 10 bằng len(CLASS_NAMES)
    SUPPORT_TENSOR = torch.zeros(1, len(CLASS_NAMES), 1, 16, 3, 224, 224).to(DEVICE)

n_way = len(CLASS_NAMES)
SUPPORT_MASK = torch.ones((1, n_way, 1, 16, 224, 224)).to(DEVICE)

def deserialize_tensor(base64_str):
    bytes_data = base64.b64decode(base64_str)
    np_arr = pickle.loads(bytes_data)
    return torch.from_numpy(np_arr).to(DEVICE)

def run_inference(video_tensor_5d):
    with torch.no_grad():
        res = model(video_tensor_5d, SUPPORT_TENSOR, SUPPORT_MASK)
        
        if isinstance(res, tuple): outputs = res[2]
        else: outputs = res
        
        scores = outputs
        if isinstance(scores, tuple): scores = scores[0]
        while scores.dim() > 2: scores = scores.mean(dim=-1)
        
        probs = torch.nn.functional.softmax(scores, dim=1)
        conf, idx = torch.max(probs, 1)
        label = CLASS_NAMES[idx.item()] if idx.item() < len(CLASS_NAMES) else "Unknown"
        
        return label, round(conf.item(), 4), scores.flatten().tolist()

for line in sys.stdin:
    try:
        parts = line.strip().split('\t')
        if len(parts) != 2: continue
        
        video_name, data_str = parts
        tensor_5d = deserialize_tensor(data_str)
        
        label, conf, logits = run_inference(tensor_5d)
        
        result = {
            "label": label, 
            "conf": conf, 
            "filename": video_name, 
            "logits": logits
        }
        print(f"{video_name}\t{json.dumps(result)}")

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
