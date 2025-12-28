import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.models.DMA.DMA import DMA
from libs.models.DMA.resnet import resnet50

class DMA_Deploy(DMA):
    def __init__(self, n_way=10, backbone='resnet50'):
        # Khởi tạo class cha nhưng tắt các tính năng thừa
        super(DMA_Deploy, self).__init__(
            n_way=n_way, 
            k_shot=1, 
            num_support_frames=16, 
            num_query_frames=16, 
            backbone=backbone,
            hidden_dim=256
        )
        
    def build_backbone(self):
        # GHI ĐÈ: Tắt pretrained=True để không download trong Docker
        if self.backbone == 'resnet50':
            self.backbone_net = resnet50(pretrained=False)
            self.layer0 = nn.Sequential(self.backbone_net.conv1, self.backbone_net.bn1, self.backbone_net.relu1, 
                                      self.backbone_net.conv2, self.backbone_net.bn2, self.backbone_net.relu2,
                                      self.backbone_net.conv3, self.backbone_net.bn3, self.backbone_net.relu3, 
                                      self.backbone_net.maxpool)
            self.layer1 = self.backbone_net.layer1
            self.layer2 = self.backbone_net.layer2
            self.layer3 = self.backbone_net.layer3
            self.layer4 = self.backbone_net.layer4
            self.feat_dim = 256
        else:
            raise ValueError("Only resnet50 supported for deploy")

    def extract_motion_features(self, s_f8, s_f4):
        # GHI ĐÈ: Sửa lỗi dimension mismatch (29 vs 28) và memory view
        B, T, C_8, H_8, W_8 = s_f8.shape
        _, _, C_4, H_4, W_4 = s_f4.shape

        m_f8 = s_f8[:, 1:] - s_f8[:, :-1]
        m_f4 = s_f4[:, 1:] - s_f4[:, :-1]
        
        # 1. Dùng reshape thay vì view để tránh lỗi non-contiguous
        m_f8 = m_f8.reshape(B, T-1, C_8, H_8, W_8).permute(0, 2, 1, 3, 4)
        m_f4 = m_f4.reshape(B, T-1, C_4, H_4, W_4).permute(0, 2, 1, 3, 4)
        
        enhanced_m_f4 = self.motion_conv_f4(m_f4)
        
        # 2. FIX QUAN TRỌNG: Resize feature map nếu lệch kích thước (29 -> 28)
        if enhanced_m_f4.shape[-2:] != m_f8.shape[-2:]:
            # Dùng trilinear interpolation cho khối 3D (Time, Height, Width)
            target_h, target_w = m_f8.shape[-2:]
            enhanced_m_f4 = F.interpolate(
                enhanced_m_f4, 
                size=(enhanced_m_f4.shape[2], target_h, target_w), 
                mode='trilinear', 
                align_corners=False
            )

        enhanced_m_f8 = self.motion_conv_f8(enhanced_m_f4 + m_f8)
        f_motion = self.motion_linear(enhanced_m_f8).sum(dim=(3,4))
        
        f_motion = f_motion.permute(0, 2, 1)
        f_motion = f_motion.reshape(B, T-1, -1)
        return f_motion