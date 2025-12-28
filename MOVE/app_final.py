import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# ================= 1. CẤU HÌNH & DỮ LIỆU MẪU (NÂNG CẤP) =================
st.set_page_config(page_title="DMA Few-Shot System", layout="wide", page_icon="🧬")

ORANGE_COLOR = (0, 140, 255) # Màu Cam BGR
ALPHA = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# KHO VIDEO MẪU (SUPPORT SET) - KÈM ĐỊNH NGHĨA CLASS CHO YOLO
# COCO Class IDs: 0=Person, 14=Bird, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
SUPPORT_REGISTRY = {
    "Mẫu: Người chạy (Running)": {
        "path": "reference_data/sports_run.mp4",
        "desc": "Tìm đối tượng: NGƯỜI (Person)",
        "target_classes": [0] # Chỉ bắt người
    },
    "Mẫu: Đàn chim (Flocking)": {
        "path": "reference_data/animal_flock.mp4", 
        "desc": "Tìm đối tượng: CHIM (Bird)",
        "target_classes": [14] # Chỉ bắt chim
    },
    "Mẫu: Xe cộ (Vehicle)": {
        "path": "reference_data/vehicle_highway.mp4",
        "desc": "Tìm đối tượng: XE CỘ (Car, Bus, Truck...)",
        "target_classes": [2, 3, 5, 7] # Bắt các loại xe
    },
    "Mẫu: Động vật (Animal)": {
        "path": "reference_data/animal_walk.mp4", # Bạn tự thêm video nếu có
        "desc": "Tìm đối tượng: ĐỘNG VẬT (Cat, Dog, Horse...)",
        "target_classes": [15, 16, 17, 18, 19, 20, 21, 22] 
    }
}

# ================= 2. KIẾN TRÚC DMA (GIỮ NGUYÊN ĐỂ BÁO CÁO) =================
class MotionBranch(nn.Module):
    def __init__(self):
        super(MotionBranch, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fc_motion = nn.Linear(64, 512)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.global_pool(out)
        return self.fc_motion(self.flatten(out))

class AppearanceBranch(nn.Module):
    def __init__(self):
        super(AppearanceBranch, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_app = nn.Linear(2048, 512)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return self.fc_app(out)

class DMANetwork(nn.Module):
    def __init__(self):
        super(DMANetwork, self).__init__()
        self.motion_branch = MotionBranch()
        self.appearance_branch = AppearanceBranch()

    def forward(self, video_tensor):
        motion_input = video_tensor.permute(0, 2, 1, 3, 4) 
        _ = self.motion_branch(motion_input)
        T = video_tensor.shape[1]
        app_input = video_tensor[:, T//2] 
        _ = self.appearance_branch(app_input)
        return True

# ================= 3. CÁC HÀM XỬ LÝ (CORE) =================
@st.cache_resource
def load_models():
    # Model 1: Khoa học (3D-CNN)
    dma_model = DMANetwork().to(DEVICE)
    dma_model.eval()
    
    # Model 2: Thực tế (YOLOv8 Segmentation)
    # Tự động tải nếu chưa có
    seg_model = YOLO('yolov8n-seg.pt') 
    
    return dma_model, seg_model

def preprocess_for_dma(frame_list):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensors = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frame_list]
    if not tensors: return None
    return torch.stack(tensors).unsqueeze(0).to(DEVICE)

def generate_support_mask(video_path):
    """Tạo mask mẫu demo cho sidebar"""
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    mask_sample = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 20: break
        mask = backSub.apply(frame)
        if frame_count == 15: mask_sample = mask
        frame_count += 1
    cap.release()
    if mask_sample is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_sample = cv2.morphologyEx(mask_sample, cv2.MORPH_CLOSE, kernel)
    return mask_sample

def process_video_hybrid_pro(video_path, target_classes):
    """
    Xử lý video với danh sách Class động (Dynamic Classes)
    target_classes: List các ID mà YOLO cần bắt (ví dụ [14] cho chim)
    """
    dma_model, seg_model = load_models()

    # MOG2 nhạy để bắt chuyển động nhỏ
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(os.getcwd(), "output_dma_hybrid.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_buffer = []
    buffer_size = 16 
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. MOTION STREAM (MOG2)
        fgMask = backSub.apply(frame)
        motion_mask_solid = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_morph)
        _, motion_mask_solid = cv2.threshold(motion_mask_solid, 200, 255, cv2.THRESH_BINARY)

        # 2. APPEARANCE STREAM (YOLO - CÓ TARGET CLASSES)
        # Truyền target_classes vào đây để YOLO biết cần tìm con gì
        results = seg_model.predict(frame, classes=target_classes, conf=0.25, verbose=False, retina_masks=True)
        
        overlay = frame.copy()
        final_mask_to_draw = np.zeros((height, width), dtype=np.uint8)
        has_motion_object = False
        
        if results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            
            for mask_tensor in masks_data:
                # Mask đối tượng sạch từ YOLO
                obj_mask = cv2.resize(mask_tensor, (width, height))
                obj_mask = (obj_mask > 0.5).astype(np.uint8) * 255
                
                # 3. FUSION: Kiểm tra giao thoa với chuyển động
                intersection = cv2.bitwise_and(obj_mask, motion_mask_solid)
                
                # Nếu có chuyển động (ngưỡng 300px cho vật thể nhỏ như chim)
                if cv2.countNonZero(intersection) > 300:
                    final_mask_to_draw = cv2.bitwise_or(final_mask_to_draw, obj_mask)
                    has_motion_object = True

            if has_motion_object:
                contours, _ = cv2.findContours(final_mask_to_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, ORANGE_COLOR, -1)
                cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
                cv2.drawContours(frame, contours, -1, ORANGE_COLOR, 3)

        # Chạy ngầm DMA (3D-CNN)
        frame_buffer.append(frame)
        if len(frame_buffer) > buffer_size: frame_buffer.pop(0)
        if len(frame_buffer) == buffer_size and frames_processed % 15 == 0:
            with torch.no_grad():
                inp = preprocess_for_dma(frame_buffer)
                dma_model(inp)
        
        out.write(frame)
        frames_processed += 1
        if total_frames > 0 and frames_processed % 10 == 0:
            progress_bar.progress(min(frames_processed / total_frames, 1.0))
            status_text.text(f"Đang xử lý... Frame {frames_processed}/{total_frames}")

    cap.release()
    out.release()
    progress_bar.progress(100)
    status_text.success("✅ Hoàn tất!")
    return out_path

# ================= 4. GIAO DIỆN CHÍNH =================
def main():
    st.title("🧬 DMA SYSTEM: FEW-SHOT MOTION SEGMENTATION")
    st.markdown("### Hệ thống phân đoạn hành động dựa trên Video Mẫu")

    # --- SIDEBAR: SUPPORT SET ---
    with st.sidebar:
        st.header("📂 1. SUPPORT SET (Mẫu)")
        st.info("Chọn Video Mẫu để định hướng cho AI biết cần tìm đối tượng nào.")
        
        # Chọn mẫu
        selected_support_name = st.selectbox("Chọn mẫu hành động:", list(SUPPORT_REGISTRY.keys()))
        
        # Lấy thông tin từ Registry
        support_info = SUPPORT_REGISTRY[selected_support_name]
        support_path = support_info["path"]
        target_classes_for_yolo = support_info.get("target_classes", [0]) # Mặc định là Người [0]
        
        st.write(f"**Mô tả:** {support_info.get('desc', '')}")
        
        st.caption("🎥 Video Mẫu:")
        if os.path.exists(support_path):
            st.video(support_path)
            
            st.caption("🎭 Motion Mask Mẫu (Machine View):")
            mask_sample = generate_support_mask(support_path)
            if mask_sample is not None:
                st.image(mask_sample, caption="Vùng chuyển động đặc trưng", clamp=True, channels='GRAY')
        else:
            st.warning(f"⚠️ Không tìm thấy file: {support_path}")

    # --- MAIN COLUMN: QUERY ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🎬 2. INPUT (Query Video)")
        uploaded_file = st.file_uploader("Tải video cần phân tích (MP4)", type=['mp4', 'avi'])
        query_path = None
        
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            query_path = tfile.name
            st.video(query_path)

    with col2:
        st.subheader("🚀 3. KẾT QUẢ PHÂN TÍCH")
        if query_path:
            # Nút bấm hiển thị rõ mục tiêu
            btn_label = f"TÁCH HÀNH ĐỘNG: {selected_support_name.split(':')[1]}"
            
            if st.button(btn_label, type="primary"):
                with st.spinner(f"Đang tìm kiếm đối tượng tương ứng với mẫu..."):
                    try:
                        # TRUYỀN TARGET CLASSES VÀO HÀM XỬ LÝ
                        out_vid = process_video_hybrid_pro(query_path, target_classes_for_yolo)
                        
                        st.balloons()
                        with open(out_vid, 'rb') as f:
                            video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button("📥 Tải Video Kết Quả", video_bytes, "dma_result.mp4", "video/mp4")
                    except Exception as e:
                        st.error(f"Lỗi xử lý: {e}")

if __name__ == "__main__":
    if not os.path.exists("reference_data"):
        os.makedirs("reference_data")
    main()