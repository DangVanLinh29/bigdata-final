import torch
import torch.nn as nn
import torchvision.models as models

class MotionBranch(nn.Module):
    """
    Luồng xử lý chuyển động (Motion Branch)
    Sử dụng 3D CNN và Pooling như trong báo cáo 
    """
    def __init__(self):
        super(MotionBranch, self).__init__()
        # Kỹ thuật Differencing thực hiện bên ngoài hoặc lớp đầu tiên
        
        # Khối 3D Convolution để bắt chuyển động thời gian
        # Input: [Batch, Channel, Depth(Time), Height, Width]
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Spatio-Temporal Pooling [cite: 98]
        
        self.flatten = nn.Flatten()
        self.fc_motion = nn.Linear(128, 512) # Vector đặc trưng chuyển động Pm

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        
        # 1. Tính sai biệt (Frame Differencing) [cite: 91]
        # x[:, :, 1:] - x[:, :, :-1] : Frame sau trừ frame trước
        motion_diff = x[:, :, 1:] - x[:, :, :-1] 
        
        # 2. Qua các lớp 3D Conv & Pooling
        out = self.conv3d_1(motion_diff)
        out = self.relu(out)
        out = self.pool3d(out)
        
        out = self.conv3d_2(out)
        out = self.relu(out)
        
        # 3. Nén thông tin (Pooling)
        out = self.global_pool(out)
        
        # 4. Trả về vector đặc trưng Pm
        Pm = self.flatten(out)
        Pm = self.fc_motion(Pm)
        return Pm

class AppearanceBranch(nn.Module):
    """
    Luồng xử lý ngoại hình (Appearance Branch)
    Sử dụng 2D CNN (ResNet) [cite: 39]
    """
    def __init__(self):
        super(AppearanceBranch, self).__init__()
        # Sử dụng Backbone ResNet50
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Bỏ lớp FC cuối
        self.fc_app = nn.Linear(2048, 512) # Vector đặc trưng hình dáng Pa

    def forward(self, x):
        # x shape: [B, C, H, W] - Chỉ lấy 1 frame đại diện (FL1)
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        Pa = self.fc_app(out)
        return Pa

class DMANetwork(nn.Module):
    """
    Mô hình tổng thể DMA (Decoupled Motion-Appearance)
    Kết hợp 2 luồng và cơ chế Attention
    """
    def __init__(self):
        super(DMANetwork, self).__init__()
        self.motion_branch = MotionBranch()
        self.appearance_branch = AppearanceBranch()
        
        # Cơ chế Fusion & Attention (Demo logic)
        self.fusion_fc = nn.Linear(512 * 2, 2) 

    def forward(self, video_tensor):
        """
        Input: video_tensor có dạng [Batch, Time, Channel, Height, Width]
        Ví dụ: [1, 16, 3, 112, 112]
        """
        # --- LUỒNG 1: MOTION (3D CNN) ---
        # 3D Conv yêu cầu Input: [Batch, Channel, Time, Height, Width]
        # Nên ta cần đổi vị trí Channel (index 2) và Time (index 1)
        motion_input = video_tensor.permute(0, 2, 1, 3, 4) 
        Pm = self.motion_branch(motion_input) 
        
        # --- LUỒNG 2: APPEARANCE (2D ResNet) ---
        # ResNet yêu cầu Input: [Batch, Channel, Height, Width]
        
        # Lấy frame ở giữa (Time // 2) làm đại diện
        T = video_tensor.shape[1]
        
        # Lỗi cũ nằm ở đây: Dư lệnh .permute không cần thiết
        # FIX: Chỉ cần lấy frame ra là nó đã tự động đúng chuẩn [B, C, H, W] rồi
        app_input = video_tensor[:, T//2] 
        
        Pa = self.appearance_branch(app_input) 
        
        # --- FUSION ---
        combined = torch.cat((Pm, Pa), dim=1)
        return combined