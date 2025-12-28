import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.models.DMA.resnet import resnet50

class Baseline_Deploy(nn.Module):
    def __init__(self, n_way=10):
        super(Baseline_Deploy, self).__init__()
        self.backbone = resnet50(pretrained=False)
        
    def forward(self, query, support, support_mask=None):
        B, T, C, H, W = query.shape
        num_support = support.shape[1]
        
        # --- 1. XỬ LÝ QUERY ---
        q = query.view(-1, C, H, W)     # (B*T, C, H, W)
        q_feat = self.backbone(q)       # Output có thể là (B*T, 2048, 7, 7) HOẶC (B*T, 2048)
        
        # [FIX]: Chỉ Pool nếu output là 4 chiều (có H, W)
        if q_feat.dim() == 4:
            q_feat = F.adaptive_avg_pool2d(q_feat, (1, 1))
            q_feat = q_feat.flatten(1)  # Ép về (B*T, 2048)
            
        # Lúc này q_feat chắc chắn là (B*T, 2048)
        q_feat = q_feat.view(B, T, -1)  # (B, T, 2048)
        q_feat = q_feat.mean(dim=1)     # (B, 2048)

        # --- 2. XỬ LÝ SUPPORT ---
        s = support.view(-1, C, H, W)
        s_feat = self.backbone(s)
        
        # [FIX]: Tương tự cho Support
        if s_feat.dim() == 4:
            s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1))
            s_feat = s_feat.flatten(1)
            
        # Reshape: (B, N, K, T, 2048) -> Lấy trung bình
        s_feat = s_feat.view(B, num_support, -1, T, s_feat.size(-1))
        s_feat = s_feat.mean(dim=(2, 3)) # Mean over K-shot & Time -> (B, N, 2048)

        # --- 3. TÍNH KHOẢNG CÁCH ---
        q_feat = F.normalize(q_feat, dim=-1).unsqueeze(1)
        s_feat = F.normalize(s_feat, dim=-1)
        logits = torch.sum(q_feat * s_feat, dim=-1) * 10.0
        
        return logits