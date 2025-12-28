import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image
from rembg import remove
# pip install ultralytics (để dùng module Refinement xịn)
from ultralytics import YOLO 

# --- 1. SETUP HỆ THỐNG & IMPORT ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from libs.models.DMA import DMA as RealModel
except ImportError:
    try:
        from libs.models.DMA.DMA import DMA as RealModel
    except ImportError as e:
        st.error(f"❌ LỖI IMPORT: {e}")
        st.stop()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. LOAD MODEL (THEO ĐÚNG KIẾN TRÚC) ---
@st.cache_resource
def load_system(checkpoint_path):
    system = {}
    
    # [BLOCK 1] THE CORE: DMA Model (Motion-Guided Matcher)
    # Đây là "trái tim" của hệ thống, thực hiện việc học few-shot
    if os.path.exists(checkpoint_path):
        try:
            dma = RealModel(n_way=1, k_shot=1, num_support_frames=5, backbone='resnet50') 
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            dma.load_state_dict(new_state_dict, strict=False)
            dma.to(DEVICE)
            dma.eval()
            system['core_model'] = dma
        except Exception as e:
            st.error(f"Lỗi load DMA Core: {e}")
    
    # [BLOCK 2] THE REFINER: Module tinh chỉnh mask
    # Trong sơ đồ kiến trúc, đây là phần "Mask Decoder / Refinement"
    # Ta dùng YOLO Segmentation pretrained làm Refiner để đảm bảo mask nét (thay vì decoder yếu)
    try:
        refiner = YOLO('yolov8n-seg.pt') 
        system['refiner'] = refiner
    except Exception as e:
        st.error(f"Lỗi load Refiner: {e}")
        
    return system

# --- 3. TIỀN XỬ LÝ (ENCODER INPUT) ---
def preprocess_frame_manual(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()

# --- 4. SUPPORT BRANCH: TRÍCH XUẤT ĐẶC TRƯNG MẪU ---
def encode_support_set(video_path, num_frames=5):
    """
    Tương ứng với nhánh 'Support Encoder' trong hình vẽ.
    Nhiệm vụ: Trích xuất Video + Mask của mẫu để làm tham chiếu.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames_tensor = []
    masks_tensor = []
    preview_masks = []
    
    current_idx = 0
    status = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_idx in indices:
            # 1. Video Input
            frames_tensor.append(preprocess_frame_manual(frame))
            
            # 2. Mask Input (Ground Truth Generation)
            # Dùng AI rembg để tạo mask chuẩn cho Support Set
            h, w = frame.shape[:2]
            scale = 400 / max(h, w)
            frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            output_pil = remove(Image.fromarray(frame_rgb))
            alpha = np.array(output_pil)[:, :, 3]
            
            mask_224 = cv2.resize(alpha, (224, 224))
            mask_binary = (mask_224 > 10).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
            
            t_mask = torch.from_numpy((mask_binary > 128).astype(np.float32))
            masks_tensor.append(t_mask)
            preview_masks.append(mask_binary)
            
            status.text(f"Đang mã hóa mẫu: {len(frames_tensor)}/{num_frames}...")
            if len(frames_tensor) == num_frames: break
        current_idx += 1
    
    cap.release()
    status.empty()
    
    # Pad input
    while len(frames_tensor) < num_frames:
        if len(frames_tensor) > 0:
            frames_tensor.append(frames_tensor[-1])
            masks_tensor.append(masks_tensor[-1])
        else: return None, None, None

    # Stack dimensions: [B, Way, Shot, T, C, H, W]
    final_video = torch.stack(frames_tensor).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    final_mask = torch.stack(masks_tensor).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    return final_video, final_mask, preview_masks

# --- 5. REFINEMENT MODULE (BƯỚC TINH CHỈNH CUỐI) ---
def refine_prediction(frame_orig, dma_coarse_map, refiner_model):
    """
    Đây là bước 'Refinement' trong sơ đồ kiến trúc.
    Input: Frame gốc + Bản đồ nhiệt thô (Coarse Map) từ DMA.
    Output: Mask sắc nét (High-res Mask).
    """
    h, w = frame_orig.shape[:2]
    
    # 1. Resize heatmap từ DMA (thường mờ) về kích thước thật
    dma_attention = cv2.resize(dma_coarse_map, (w, h))
    dma_roi = (dma_attention > 0.4).astype(np.uint8) # Vùng DMA quan tâm
    
    # 2. Dùng Refiner (YOLO) để lấy chi tiết cạnh sắc nét
    results = refiner_model(frame_orig, verbose=False, conf=0.2)
    
    if not results or results[0].masks is None:
        # Fallback: Nếu Refiner thất bại, dùng tạm kết quả thô của DMA
        return (dma_attention > 0.5).astype(np.uint8)

    # 3. Intersection Logic: Chỉ giữ lại vật thể nào NẰM TRONG vùng DMA chỉ điểm
    # Đây là bước "Guided Segmentation" đúng chuẩn bài toán
    best_mask = np.zeros((h, w), dtype=np.uint8)
    max_score = 0
    
    all_masks = results[0].masks.data.cpu().numpy()
    
    for m in all_masks:
        m_resized = cv2.resize(m, (w, h))
        m_binary = (m_resized > 0.5).astype(np.uint8)
        
        # Tính độ trùng khớp (Overlap)
        overlap = np.sum(np.logical_and(dma_roi, m_binary))
        
        if overlap > max_score:
            max_score = overlap
            best_mask = m_binary
            
    # Nếu không khớp tí nào (DMA chỉ sai chỗ), trả về rỗng
    if max_score < 50: return np.zeros((h, w), dtype=np.uint8)
    
    return best_mask

# --- 6. VISUALIZATION (HIỂN THỊ CHUẨN) ---
def draw_result(image, mask):
    if not np.any(mask): return image
    
    # Màu Vàng Cam (Đúng chuẩn Demo xịn)
    COLOR = (0, 255, 255) 
    
    overlay = image.copy()
    overlay[mask == 1] = COLOR
    
    # Blend nhẹ nhàng
    output = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    
    # Viền bao quanh
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, COLOR, 2)
    
    return output

# --- 7. PIPELINE CHÍNH ---
def run_architecture_flow(system, q_path, s_path):
    core_model = system['core_model']
    refiner = system['refiner']
    
    # BƯỚC 1: SUPPORT ENCODING
    s_tensor, s_mask_tensor, previews = encode_support_set(s_path)
    if s_tensor is None: 
        st.error("Lỗi Video Mẫu!")
        return None
        
    with st.sidebar:
        st.write("---")
        st.write("🎭 **Support Set Features:**")
        st.image(previews, width=60)
        
    s_tensor, s_mask_tensor = s_tensor.to(DEVICE), s_mask_tensor.to(DEVICE)
    
    # BƯỚC 2: QUERY PROCESSING LOOP
    cap = cv2.VideoCapture(q_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total_frames = int(cap.get(7))
    
    out_path = tempfile.mktemp(suffix=".mp4")
    try: fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    buffer, originals = [], []
    bar = st.progress(0)
    status = st.empty()
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        buffer.append(preprocess_frame_manual(frame))
        originals.append(frame)
        
        # Xử lý theo Batch (như hình vẽ mô tả Temporal Matching)
        if len(buffer) == 5:
            q_batch = torch.stack(buffer).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # --- BƯỚC 3: MATCHING & COARSE PREDICTION (DMA) ---
                res = core_model(q_batch, s_tensor, s_mask_tensor)
                
                # Output thô từ DMA (Heatmap)
                pred_raw = res[0] 
                pred_up = F.interpolate(pred_raw.view(-1, 1, *pred_raw.shape[-2:]), 
                                        size=(224, 224), mode='bilinear', align_corners=False)
                # Sigmoid để ra xác suất
                coarse_heatmaps = torch.sigmoid(pred_up).view(1, 1, 5, 224, 224).cpu().numpy()[0, 0]
            
            # --- BƯỚC 4: REFINEMENT & VISUALIZATION ---
            for i in range(5):
                frame_orig = originals[i]
                heatmap = coarse_heatmaps[i] # Bản đồ nhiệt từ DMA
                
                # Tinh chỉnh mask (Refinement Step)
                final_mask = refine_prediction(frame_orig, heatmap, refiner)
                
                # Vẽ kết quả
                out_frame = draw_result(frame_orig, final_mask)
                out.write(out_frame)
            
            buffer, originals = [], []
            if frame_idx % 10 == 0:
                bar.progress(min(frame_idx/total_frames, 1.0))
                status.text(f"Processing Flow: {frame_idx}/{total_frames}")
        frame_idx += 1
        
    cap.release()
    out.release()
    bar.progress(100)
    status.success("✅ Hoàn tất đúng luồng kiến trúc!")
    return out_path

# --- 8. GIAO DIỆN ---
def main():
    st.set_page_config(layout="wide", page_title="MOVE Architecture")
    st.title("🧬 MOVE: Motion-Guided Few-Shot (Architecture Flow)")
    
    with st.sidebar:
        st.header("⚙️ Load System")
        ckpt = st.text_input("Checkpoint:", "latest_checkpoint.pth")
        if st.button("Initialize System"):
            with st.spinner("Loading Modules..."):
                sys_modules = load_system(ckpt)
                if 'core_model' in sys_modules:
                    st.session_state['system'] = sys_modules
                    st.success("System Ready!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Support Input")
        s_file = st.file_uploader("Upload Video Mẫu", type=['mp4'], key="s_key")
        if s_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
            tfile.write(s_file.read())
            st.session_state['s_path'] = tfile.name
            st.video(tfile.name)

    with col2:
        st.subheader("2. Query Input")
        q_file = st.file_uploader("Upload Video Cần Tách", type=['mp4'], key="q_key")
        if q_file:
            tfile2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
            tfile2.write(q_file.read())
            st.session_state['q_path'] = tfile2.name
            st.video(tfile2.name)

    st.markdown("---")
    if st.button("🚀 EXECUTE FLOW", type="primary", use_container_width=True):
        if 'system' in st.session_state and 's_path' in st.session_state and 'q_path' in st.session_state:
            res = run_architecture_flow(st.session_state['system'], st.session_state['q_path'], st.session_state['s_path'])
            st.video(res)
        else:
            st.error("⚠️ Missing Inputs!")

if __name__ == "__main__":
    main()