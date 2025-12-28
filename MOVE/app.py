import streamlit as st
import torch
import numpy as np
import tempfile
import os
import sys
import cv2
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import json
import warnings
import traceback

# ================= 1. CẤU HÌNH CƠ BẢN =================
warnings.filterwarnings("ignore")
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚠️ ĐƯỜNG DẪN FILE MODEL (CHECK LẠI TRÊN MÁY BẠN)
CHECKPOINT_PATH = r"C:\Nhom_3_BigData\MOVE\logs_debug\resnet50\default\2-way-2-shot\group0\latest_checkpoint.pth"

# ================= 2. DỮ LIỆU MẪU (VIDEO REGISTRY) =================
# Đây là kho dữ liệu mẫu mà MapReduce "giả lập" đã tạo ra
VIDEO_REGISTRY = {
    "reference_data/animal_flock.mp4": {"obj": "Animal", "motion": "Flocking", "desc": "Đàn chim/thú di chuyển theo bầy"},
    "reference_data/animal_walk.mp4": {"obj": "Animal", "motion": "Walking", "desc": "Động vật đang đi bộ"},
    "reference_data/sports_run.mp4":  {"obj": "Sports", "motion": "Running", "desc": "Người đang chạy"},
    "reference_data/vehicle_highway.mp4": {"obj": "Vehicle", "motion": "Fast Moving", "desc": "Phương tiện di chuyển tốc độ cao"},
    "reference_data/daily_phone.mp4": {"obj": "Daily", "motion": "Daily Activity", "desc": "Hoạt động thường ngày (Ăn/Uống/Đọc sách)"}
}

# ================= 3. HÀM TẠO THƯ MỤC & LOAD MODEL =================
def check_and_create_folders():
    if not os.path.exists("reference_data"):
        os.makedirs("reference_data")
        st.warning("⚠️ Đã tạo thư mục 'reference_data'. Hãy copy video mẫu vào đó!")
        return False
    return True

@st.cache_resource
def load_full_dma_model():
    print("🏗️ Đang khởi tạo FULL DMA NETWORK...")
    try:
        from libs.models.DMA.DMA import DMA 
    except ImportError as e:
        st.error(f"❌ Lỗi Import DMA: {e}")
        st.stop()

    try:
        model = DMA(
            n_way=1, k_shot=1,
            num_support_frames=16, num_query_frames=16,
            num_meta_motion_queries=15
        )
        
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Đã nạp weights thành công!")
        else:
            st.error(f"⚠️ Không tìm thấy file checkpoint: {CHECKPOINT_PATH}")
            st.stop()
            
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Model: {e}")
        st.stop()

# ================= 4. XỬ LÝ VIDEO & MASK =================
def process_video_5d(video_path, num_frames=16):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return None
    
    indices = np.linspace(0, total_frames-1, num_frames).astype(int)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(transform(frame))
    cap.release()
    while len(frames) < num_frames: frames.append(frames[-1])
    
    tensor = torch.stack(frames).unsqueeze(0)
    return tensor.to(DEVICE).float()

def create_dummy_mask(shape):
    return torch.ones(shape).float().to(DEVICE)

# ================= 5. HÀM SUY LUẬN (CORE) =================
def run_dma_inference(model, query_path, support_path):
    if not os.path.exists(support_path): return 0.0

    q_tensor_raw = process_video_5d(query_path) 
    s_tensor_raw = process_video_5d(support_path) 
    
    if q_tensor_raw is None or s_tensor_raw is None: return 0.0

    try:
        # Reshape cho khớp Model (Support 7D, Query 5D)
        s_tensor = s_tensor_raw.view(1, 1, 1, 16, 3, 224, 224)
        q_tensor = q_tensor_raw 
        s_mask = create_dummy_mask((1, 1, 1, 16, 224, 224))
        q_mask_dummy = create_dummy_mask((1, 1, 16, 224, 224))

        with torch.no_grad():
            output = model(q_tensor, s_tensor, s_mask, q_mask_dummy)
            
            pred_mask = None
            if isinstance(output, dict):
                pred_mask = output.get('pred_masks', output.get('masks', list(output.values())[0]))
            elif isinstance(output, (list, tuple)):
                pred_mask = output[-1]
            else:
                pred_mask = output

            if pred_mask is not None:
                probs = torch.sigmoid(pred_mask) if pred_mask.min() < 0 else pred_mask
                score = torch.mean(torch.topk(probs.flatten(), k=max(1, int(probs.numel()*0.1))).values)
                return score.item()
            return 0.0

    except Exception as e:
        print(f"Lỗi Inference: {e}")
        return 0.0

# ================= 6. GIAO DIỆN WEB (JSON OUTPUT) =================
st.set_page_config(page_title="DMA BIG DATA SYSTEM", layout="wide", page_icon="🧬")

st.title("🧬 DMA SYSTEM: DECOUPLED MOTION-APPEARANCE")
st.markdown("### Hệ thống Xử lý Video Phân tán & Nhận diện Hành động")
st.markdown("---")

check_and_create_folders()
model = load_full_dma_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Query Video")
    uploaded_file = st.file_uploader("Tải video lên", type=['mp4', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close() 
        st.video(tfile.name)

with col2:
    st.subheader("2. Analysis Result (JSON)")
    
    if uploaded_file:
        if st.button("🚀 CHẠY MAP-REDUCE & PREDICT"):
            with st.spinner("Đang trích xuất đặc trưng & so khớp..."):
                
                best_score = -1.0
                best_match = None
                
                # Thanh tiến trình
                prog_bar = st.progress(0)
                items = list(VIDEO_REGISTRY.items())
                
                for idx, (ref_path, meta) in enumerate(items):
                    if not os.path.exists(ref_path): continue
                    
                    score = run_dma_inference(model, tfile.name, ref_path)
                    
                    if score > best_score:
                        best_score = score
                        best_match = meta
                    
                    prog_bar.progress((idx + 1) / len(items))

                st.write("---")
                
                # --- PHẦN HIỂN THỊ KẾT QUẢ DẠNG JSON ---
                if best_match and best_score > 0.0:
                    st.success("✅ Đã nhận diện thành công!")
                    
                    # Tạo cấu trúc JSON kết quả
                    result_json = {
                        "object_name": best_match['obj'],      # Tên đối tượng (VD: Animal)
                        "action_name": best_match['motion'],   # Tên hành động (VD: Flocking)
                        "confidence_score": round(best_score, 4), # Điểm tin cậy
                        "description": best_match['desc'],     # Mô tả
                        "system_status": "Success",
                        "processing_mode": "DMA_Inference"
                    }
                    
                    # Hiển thị JSON đẹp
                    st.json(result_json)
                    
                else:
                    st.error("Không tìm thấy kết quả phù hợp.")
                    # JSON lỗi
                    st.json({
                        "status": "Failed",
                        "reason": "Low confidence or missing reference data"
                    })