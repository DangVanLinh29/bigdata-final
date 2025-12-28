import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.DMA.resnet import resnet50
from libs.models.DMA.transformers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from libs.models.DMA.video_swin_referfomer.video_swin_transformer import VideoSwinTransformerBackbone


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class DMA(nn.Module):
    """Decoupled Motion-Appearance Network"""
    def __init__(self, 
                 n_way=1,
                 k_shot=1, 
                 num_support_frames=5,
                 num_query_frames=1,
                 num_meta_motion_queries=15,
                 backbone='resnet50',
                 hidden_dim=256,
                 num_q_former_layers=6,
                 motion_appear_orth=False):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_support_frames = num_support_frames
        self.num_query_frames = num_query_frames
        self.backbone = backbone
        self.motion_appear_ortho = motion_appear_orth
    
        self.build_backbone()
        # lateral connections for feature pyramid
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        # feature dimensions for ResNet50
        self.in_features = ["res2", "res3", "res4", "res5"]
        if backbone == 'resnet50':
            self.in_channels = [256, 512, 1024, 2048]
        elif backbone == 'videoswin':
            self.in_channels = [96, 192, 384, 768]
        self.feat_dim = 256
        # feature dimensions
        self.hidden_dim = hidden_dim
        
        # Build lateral and output convolutions for FPN
        # Lateral convs reduce channel dimensions to hidden_dim
        # Output convs do 3x3 convolution on the merged features
        for _, in_channel in enumerate(self.in_channels):
            lateral_conv = nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1)
            output_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
            
            # Initialize weights using Xavier initialization
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
            
        # ========================================================
        # q-former
        self.num_meta_motion_queries = num_meta_motion_queries
        self.num_q_former_layers = num_q_former_layers
        
        # ========================================================
        # segmentation head
        # self.decoder = MaskFormerDecoder(self.hidden_dim)
        self.proposal_generator = MaskProposalGeneratorAdj(self.hidden_dim)
        
        self.motion_prototype_network = MotionProtoptyeNetwork(self.hidden_dim, self.num_meta_motion_queries, self.num_q_former_layers)
        self.prototype_enhancer = PrototypeEnhancer(self.hidden_dim)
        
        self.motion_aware_decoder = MotionAwareDecoder(self.hidden_dim)
        self.motion_cls_norm = nn.LayerNorm(self.hidden_dim)
        # self.motion_cls_head = nn.Linear(self.hidden_dim, 1)
        # ======================================================== 

        # Category classification heads
        if self.motion_appear_ortho:
            self.category_classifier = nn.Linear(self.hidden_dim, 88)  # 88 object categories
            self.motion_classifier = nn.Linear(self.hidden_dim, 224)   # 224 motion categories

        # motion feature extraction: lightweight convolution
        self.motion_conv_f4 = nn.Sequential(
            nn.Conv3d(256, 256, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, (1, 2, 2), padding=(0, 1, 1), stride=(1, 2, 2))
        )
        self.motion_conv_f8 = nn.Sequential(
            nn.Conv3d(self.hidden_dim, self.hidden_dim, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.hidden_dim),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, (1, 2, 2), padding=(0, 1, 1), stride=(1, 2, 2))
        )
        self.motion_linear = nn.Conv3d(self.hidden_dim, self.hidden_dim, (1, 1, 1))

    @property
    def device(self):
        return next(self.parameters()).device

    def build_backbone(self):
        """Build backbone network"""
        # Load pretrained ResNet50
        if self.backbone == 'resnet50':
            backbone = resnet50(pretrained=True)
            # Get features from different layers
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.conv2, backbone.bn2, backbone.relu2,
                                        backbone.conv3, backbone.bn3, backbone.relu3, backbone.maxpool)
            self.layer1 = backbone.layer1  # 1/4
            self.layer2 = backbone.layer2  # 1/8 
            self.layer3 = backbone.layer3  # 1/16
            self.layer4 = backbone.layer4  # 1/32
            
            # Set feature dimensions for each layer
            self.feat_dims = {
                'layer1': 256,   # layer1 output
                'layer2': 512,   # layer2 output  
                'layer3': 1024, # layer3 output
                'layer4': 2048  # layer4 output
            }
            
            self.feat_dim = 256  # Keep main feature dimension unchanged
        elif self.backbone == 'videoswin':
            configs = {
                'video_swin_t_p4w7':
                            dict(patch_size=(1,4,4),
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=(8,7,7),
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.2,
                                    patch_norm=True,
                                    use_checkpoint=False
                                    ),
            }
            print('Building video swin backbone from swin_tiny_patch244_window877_kinetics400_1k.pth')
            cfgs = configs['video_swin_t_p4w7']
            self.encoder = VideoSwinTransformerBackbone(True, './pretrain_model/swin_tiny_patch244_window877_kinetics400_1k.pth', True, **cfgs)
            print('Building backbone successfully!')
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
    def extract_features(self, rgb_input, mask_feats=False, num_frames=None):
        """Extract multi-scale features using backbone network"""
        # Extract multi-scale features
        if self.backbone == 'resnet50':
            feat_1_4 = self.layer1(self.layer0(rgb_input))
            feat_1_8 = self.layer2(feat_1_4)
            feat_1_16 = self.layer3(feat_1_8)
            feat_1_32 = self.layer4(feat_1_16)
        elif self.backbone == 'videoswin':
            assert num_frames is not None
            feats = self.encoder(rgb_input, num_frames)
            feat_1_4 = feats['0']
            feat_1_8 = feats['1']
            feat_1_16 = feats['2']
            feat_1_32 = feats['3']
            
            # Return multi-scale feature dictionary
        features = {
            'res2': feat_1_4,
            'res3': feat_1_8,
            'res4': feat_1_16,
            'res5': feat_1_32,
        }
            
        if mask_feats:
            # Generate multi-scale features using FPN
            ms_feats = []
            
            # Start from highest level (1/32 scale)
            prev_features = self.lateral_convs[3](features['res5'])
            ms_feats.append(self.output_convs[3](prev_features))
            
            # Generate 1/16 scale features
            p4 = self.lateral_convs[2](features['res4'])
            p4_up = F.interpolate(prev_features, size=features['res4'].shape[-2:], 
                                mode='bilinear', align_corners=False)
            prev_features = p4 + p4_up
            ms_feats.append(self.output_convs[2](prev_features))
            
            # Generate 1/8 scale features
            p3 = self.lateral_convs[1](features['res3'])
            p3_up = F.interpolate(prev_features, size=features['res3'].shape[-2:],
                                mode='bilinear', align_corners=False)
            prev_features = p3 + p3_up
            ms_feats.append(self.output_convs[1](prev_features))
            
            # Generate 1/4 scale features
            p2 = self.lateral_convs[0](features['res2'])
            p2_up = F.interpolate(prev_features, size=features['res2'].shape[-2:],
                                mode='bilinear', align_corners=False)
            prev_features = p2 + p2_up
            ms_feats.append(self.output_convs[0](prev_features))
    
        features['ms_feats'] = ms_feats
        
        return features
    
    def mask_pooling(self, mask_feats, mask):
        # mask_feats: (B, c, h1, w1)
        # mask: (B, h2, w2)
        # output: (B, C)
        B = mask.shape[0]
        
        # Resize mask to match mask_feats spatial dimensions
        if mask.shape[-2:] != mask_feats.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=mask_feats.shape[-2:], 
                               mode='bilinear', align_corners=False).squeeze(1)
        
        mask = mask.view(B, -1)
        # Use mask pooling to extract foreground features
        mask_feats = mask_feats.view(B, -1, mask.shape[-1])
        mask_feats = mask_feats * mask.unsqueeze(1)
        mask_feats = mask_feats.sum(dim=-1) / (mask.sum(dim=-1, keepdim=True) + 1e-6)
        
        return mask_feats
    
    def forward(self, query_video, support_video, support_mask, query_mask=None, require_cls=False, oracle_mask=False):
        if self.training:
            return self.forward_train(query_video, support_video, support_mask, query_mask)
        else:
            return self.forward_test(query_video, support_video, support_mask, query_mask, require_cls=require_cls, oracle_mask=oracle_mask)
    
    def extract_motion_features(self, s_f8, s_f4):
        B, T, C_8, H_8, W_8 = s_f8.shape
        _, _, C_4, H_4, W_4 = s_f4.shape
        

        m_f8 = s_f8[:, 1:] - s_f8[:, :-1]
        m_f4 = s_f4[:, 1:] - s_f4[:, :-1]
        # [B, n_way, k_shot, s_frames-1, C, H, W]
        m_f8 = m_f8.view(B, T-1, C_8, H_8, W_8).permute(0, 2, 1, 3, 4)
        m_f4 = m_f4.view(B, T-1, C_4, H_4, W_4).permute(0, 2, 1, 3, 4)
        enhanced_m_f4 = self.motion_conv_f4(m_f4)
        # ================= [FIX LỖI 5D TENSOR RESIZE] =================
        # Kiểm tra kích thước H, W có lệch nhau không
        if enhanced_m_f4.shape[-2:] != m_f8.shape[-2:]:
            target_size = m_f8.shape[-2:] # Lấy kích thước chuẩn (28, 28)
            
            # TRƯỜNG HỢP 1: Dữ liệu 5 Chiều (Batch, Channel, Time, H, W) -> Video
            if enhanced_m_f4.dim() == 5:
                B, C, T, H, W = enhanced_m_f4.shape
                # "Dát mỏng": Gộp Batch và Time lại -> (B*T, C, H, W)
                temp_input = enhanced_m_f4.view(-1, C, H, W)
                
                # Co kéo (Interpolate) trên không gian 2D
                temp_output = torch.nn.functional.interpolate(
                    temp_input, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # "Xếp chồng" lại: Tách Batch và Time ra -> (B, C, T, H_new, W_new)
                enhanced_m_f4 = temp_output.view(B, C, T, target_size[0], target_size[1])
            
            # TRƯỜNG HỢP 2: Dữ liệu 4 Chiều (Batch, Channel, H, W) -> Ảnh thường
            else:
                enhanced_m_f4 = torch.nn.functional.interpolate(
                    enhanced_m_f4, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # Cộng sau khi đã khớp kích thước
        enhanced_m_f8 = self.motion_conv_f8(enhanced_m_f4 + m_f8)
        # =============================================================
        f_motion = self.motion_linear(enhanced_m_f8).sum(dim=(3,4))
        f_motion = f_motion.permute(0, 2, 1)
        f_motion = f_motion.view(B, -1, f_motion.size(-1))
        return f_motion
        
    def forward_train(self, query_video, support_video, support_mask, query_mask=None):
        # Shape check
        B, T, C, H, W = query_video.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert support_video.shape == (B, self.n_way, self.k_shot, self.num_support_frames, 3, H, W)
        assert support_mask.shape == (B, self.n_way, self.k_shot, self.num_support_frames, H, W)
        assert query_mask.shape == (B, self.n_way, T, H, W)
        
        if self.backbone == 'resnet50':
            query_features = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True)
            support_features = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True)
        elif self.backbone == 'videoswin':
            query_features = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True, num_frames=self.num_query_frames)
            support_features = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True, num_frames=self.num_support_frames)
    
        proposal_masks = []
        
        predict_mask_list = []
        predict_cls_list = []
        
        gt_mask_list = []
        gt_cls_list = []

        all_gt_mask = []
        for b in range(B):
            valid_gt_mask = []
            for N_way in range(self.n_way):
                if torch.any(query_mask[b, N_way] > 0):
                    valid_gt_mask.append(query_mask[b, N_way])
            if len(valid_gt_mask) == 0:
                valid_gt_mask.append(query_mask[b, 0])
            while len(valid_gt_mask) < self.n_way:
                valid_gt_mask.append(valid_gt_mask[-1])
            all_gt_mask.append(torch.stack(valid_gt_mask, dim=0))
        all_gt_mask = torch.stack(all_gt_mask, dim=0)
        
        support_motion_prototypes = []
        support_motion_pe = []
        support_category_logits = []
        support_motion_logits = []
        orthogonal_loss_list = 0.
        for N_way in range(self.n_way):
            # Process each way separately
            s_f32, s_f16, s_f8, s_f4 = support_features['ms_feats']
            s_f32 = s_f32.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f32.shape[-2:])
            s_f16 = s_f16.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f16.shape[-2:])
            s_f8 = s_f8.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f8.shape[-2:])
            s_f4 = s_f4.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f4.shape[-2:])
            s_f_motion = self.extract_motion_features(s_f8=s_f8[:, N_way].flatten(0, 1), s_f4=s_f4[:, N_way].flatten(0, 1))
            s_f_motion = s_f_motion.view(B, self.k_shot, self.num_support_frames - 1, self.hidden_dim)
            
            s_f32 = s_f32[:, N_way].flatten(0, 2)
            s_f16 = s_f16[:, N_way].flatten(0, 2)
            s_f8 = s_f8[:, N_way].flatten(0, 2)
            s_f4 = s_f4[:, N_way].flatten(0, 2)
            
            s_mask = support_mask[:, N_way].flatten(0, 2)
            # gt_query_mask = query_mask[:, N_way].flatten(0, 1)
            gt_query_mask = all_gt_mask[:, N_way].flatten(0, 1)

            # support prototype
            s_mask_feats = self.mask_pooling(s_f4, s_mask)
            
            # Category classification from mask features
            if self.motion_appear_ortho:
                s_mask_feats_avg = s_mask_feats.reshape(B, self.num_support_frames, -1).mean(dim=1)  
                s_category_logits = self.category_classifier(s_mask_feats_avg)
                support_category_logits.append(s_category_logits)
                
                # Motion classification from motion features
                s_motion_feats_avg = s_f_motion.reshape(B, -1, self.hidden_dim).mean(dim=1)
                s_motion_logits = self.motion_classifier(s_motion_feats_avg)
                support_motion_logits.append(s_motion_logits)
                
                # Compute orthogonality between motion and appearance features
                s_f_motion_flat = s_f_motion.reshape(-1, self.hidden_dim)  # Flatten to (B*K*T, hidden_dim)
                s_mask_feats_flat = s_mask_feats.reshape(-1, self.hidden_dim)  # Flatten to (B*K*T, hidden_dim)
                
                # Normalize features
                s_f_motion_norm = F.normalize(s_f_motion_flat, p=2, dim=1)
                s_mask_feats_norm = F.normalize(s_mask_feats_flat, p=2, dim=1)
                
                # Compute cosine similarity
                similarity = torch.mm(s_f_motion_norm, s_mask_feats_norm.t())
                
                # Orthogonality loss - minimize cosine similarity
                orthogonal_loss = torch.mean(torch.abs(similarity))
                orthogonal_loss_list += orthogonal_loss
            
            s_motion_p, s_motion_pe = self.motion_prototype_network(s_mask_feats.view(B, self.k_shot, self.num_support_frames, -1), s_f_motion, self.k_shot, self.num_support_frames)
            support_motion_prototypes.append(s_motion_p)
            support_motion_pe.append(s_motion_pe)
            
            # predict_query_mask
            q_f32, q_f16, q_f8, q_f4 = query_features['ms_feats']
            q_f_motion = self.extract_motion_features(s_f8=q_f8.view(B, self.num_query_frames, *q_f8.shape[-3:]), s_f4=q_f4.view(B, self.num_query_frames, *q_f4.shape[-3:]))
            q_f_motion = q_f_motion.unsqueeze(1)
            query_proposal_mask = self.proposal_generator(q_f32, q_f16, q_f8)  # Need to add a motion_aware module
            
            # gt_query_mask motion prototype
            gt_query_mask_feats = self.mask_pooling(q_f4, gt_query_mask)
            gt_query_motion_p, gt_query_motion_pe = self.motion_prototype_network(gt_query_mask_feats.view(B, 1, T, -1), q_f_motion, 1, T)
            enhance_gt_query_motion_p, enhance_gt_query_motion_pe, gt_motion_cls = self.prototype_enhancer(gt_query_motion_p, gt_query_motion_pe, s_motion_p, s_motion_pe)
            
            gt_proposal_mask = F.interpolate(
                gt_query_mask.unsqueeze(1),
                size = q_f8.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # gt_output_mask = self.motion_aware_decoder(enhance_gt_query_motion_p,
            #                         enhance_gt_query_motion_pe, 
            #                         q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
            #                         q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
            #                         q_f4.view(B, T, -1, *q_f4.shape[-2:]),
            #                         gt_proposal_mask,)
            gt_output_mask = self.motion_aware_decoder(enhance_gt_query_motion_p,
                                    enhance_gt_query_motion_pe, 
                                    q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
                                    q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
                                    q_f4.view(B, T, -1, *q_f4.shape[-2:]),
                                    )
            
            gt_mask_list.append(gt_output_mask)
            # gt_cls_list.append(self.motion_cls_head(gt_motion_cls))
            gt_cls_list.append(gt_motion_cls)
            
            # motion prototype
            q_mask_feats = self.mask_pooling(q_f4, query_proposal_mask.sigmoid())
            q_motion_p, q_motion_pe = self.motion_prototype_network(q_mask_feats.view(B, 1, T, -1), q_f_motion, 1, T)

            enhance_q_motion_p, enhance_q_motion_pe, q_motion_cls = self.prototype_enhancer(q_motion_p, q_motion_pe, s_motion_p, s_motion_pe)
            
            # motion aware decoder
            # mask = self.motion_aware_decoder(enhance_q_motion_p, 
            #                                  enhance_q_motion_pe, 
            #                                  q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
            #                                  q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
            #                                  q_f4.view(B, T, -1, *q_f4.shape[-2:]),
            #                                  query_proposal_mask,)
            mask = self.motion_aware_decoder(enhance_q_motion_p, 
                                             enhance_q_motion_pe, 
                                             q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
                                             q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
                                             q_f4.view(B, T, -1, *q_f4.shape[-2:]),
                                             )
            
            # q_motion_cls = self.motion_cls_head(q_motion_cls)
            predict_cls_list.append(q_motion_cls)
            predict_mask_list.append(mask)
            
            
            proposal_masks.append(query_proposal_mask.view(B, T, *query_proposal_mask.shape[-2:]))
            
        predict_mask = torch.stack(predict_mask_list, dim=1)
        predict_cls = torch.stack(predict_cls_list, dim=1)
        
        gt_mask = torch.stack(gt_mask_list, dim=1)
        gt_cls = torch.stack(gt_cls_list, dim=1)
        
        proposal_masks = torch.stack(proposal_masks, dim=1)
        
        support_motion_prototype_1 = support_motion_prototypes[0]
        support_motion_pe_1 = support_motion_pe[0]
        support_motion_prototype_2 = support_motion_prototypes[1]
        support_motion_pe_2 = support_motion_pe[1]
        
        _, _, gt_motion_cls_1 = self.prototype_enhancer(support_motion_prototype_1, support_motion_pe_1, support_motion_prototype_2, support_motion_pe_2)
        _, _, gt_motion_cls_2 = self.prototype_enhancer(support_motion_prototype_2, support_motion_pe_2, support_motion_prototype_1, support_motion_pe_1)
        # gt_motion_cls_1 = self.motion_cls_head(gt_motion_cls_1)
        # gt_motion_cls_2 = self.motion_cls_head(gt_motion_cls_2)
        
        support_output_motion_cls = torch.stack([gt_motion_cls_1, gt_motion_cls_2], dim=1)
        support_gt_motion_cls = torch.zeros_like(support_output_motion_cls)[..., 0]
        
        # Prepare category classification results
        if self.motion_appear_ortho:
            support_category_logits = torch.stack(support_category_logits, dim=1)  # [B, N_way, 73]
            support_motion_logits = torch.stack(support_motion_logits, dim=1)      # [B, N_way, 224]
            return predict_mask, predict_cls, gt_mask, gt_cls, proposal_masks, support_output_motion_cls, support_gt_motion_cls, enhance_q_motion_p, enhance_gt_query_motion_p, support_category_logits, support_motion_logits, orthogonal_loss_list
        else:
            return predict_mask, predict_cls, gt_mask, gt_cls, proposal_masks, support_output_motion_cls, support_gt_motion_cls, enhance_q_motion_p, enhance_gt_query_motion_p

    def forward_test(self, query_video, support_video, support_mask, query_mask=None, require_cls=False, oracle_mask=False):
        """
        The parameter require_cls and oracle_mask only used for visualization and ablation study
        Please dont use them during normally training and inference. And remove them in the version for release
        """
        # Shape check
        B, T, C, H, W = query_video.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert support_video.shape == (B, self.n_way, self.k_shot, self.num_support_frames, 3, H, W)
        assert support_mask.shape == (B, self.n_way, self.k_shot, self.num_support_frames, H, W)


        if query_mask is not None:
            assert (B, self.n_way, T, H, W) == query_mask.shape
        
        if self.backbone == 'resnet50':
            query_features = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True)
            support_features = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True)
        elif self.backbone == 'videoswin':
            query_features = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True, num_frames=self.num_query_frames)
            support_features = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True, num_frames=self.num_support_frames)
        
        N_way_mask = []
        proposal_masks = []
        motion_cls = []
        for N_way in range(self.n_way):
            # Process each way separately
            s_f32, s_f16, s_f8, s_f4 = support_features['ms_feats']
            s_f32 = s_f32.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f32.shape[-2:])
            s_f16 = s_f16.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f16.shape[-2:])
            s_f8 = s_f8.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f8.shape[-2:])
            s_f4 = s_f4.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f4.shape[-2:])
            s_f_motion = self.extract_motion_features(s_f8=s_f8[:, N_way].flatten(0, 1), s_f4=s_f4[:, N_way].flatten(0, 1))
            s_f_motion = s_f_motion.view(B, self.k_shot, self.num_support_frames - 1, self.hidden_dim)
            
            s_f32 = s_f32[:, N_way].flatten(0, 2)
            s_f16 = s_f16[:, N_way].flatten(0, 2)
            s_f8 = s_f8[:, N_way].flatten(0, 2)
            s_f4 = s_f4[:, N_way].flatten(0, 2)
            
            s_mask = support_mask[:, N_way].flatten(0, 2)
            
            # support prototype
            s_mask_feats = self.mask_pooling(s_f4, s_mask)
            s_motion_p, s_motion_pe = self.motion_prototype_network(s_mask_feats.view(B, self.k_shot, self.num_support_frames, -1), s_f_motion, self.k_shot, self.num_support_frames)
            
            q_f32, q_f16, q_f8, q_f4 = query_features['ms_feats']
            q_f_motion = self.extract_motion_features(s_f8=q_f8.view(B, T, *q_f8.shape[-3:]), s_f4=q_f4.view(B, T, *q_f4.shape[-3:]))
            q_f_motion = q_f_motion.unsqueeze(1)
            query_proposal_mask = self.proposal_generator(q_f32, q_f16, q_f8)  # Need to add a motion_aware module

            if oracle_mask:
                query_proposal_mask = F.interpolate(
                    query_mask[:, N_way],
                    size = query_proposal_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )[0]
            
            # motion prototype
            q_mask_feats = self.mask_pooling(q_f4, query_proposal_mask.sigmoid())
            q_motion_p, q_motion_pe = self.motion_prototype_network(q_mask_feats.view(B, 1, T, -1), q_f_motion, 1, T)
            
            if not require_cls:
                enhance_q_motion_p, enhance_q_motion_pe, q_motion_cls = self.prototype_enhancer(q_motion_p, q_motion_pe, s_motion_p, s_motion_pe)
            else:
                enhance_q_motion_p, enhance_q_motion_pe, q_motion_cls, q_cls_vec, s_cls_vec = \
                    self.prototype_enhancer(q_motion_p, q_motion_pe, s_motion_p, s_motion_pe, require_cls=True)
            
            # motion aware decoder
            # mask = self.motion_aware_decoder(enhance_q_motion_p, 
            #                                  enhance_q_motion_pe, 
            #                                  q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
            #                                  q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
            #                                  q_f4.view(B, T, -1, *q_f4.shape[-2:]),
            #                                  query_proposal_mask,
            #                                  q_f_motion)
            mask = self.motion_aware_decoder(enhance_q_motion_p, 
                                             enhance_q_motion_pe, 
                                             q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
                                             q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
                                             q_f4.view(B, T, -1, *q_f4.shape[-2:]),
                                             )
            
            # q_motion_cls = self.motion_cls_head(q_motion_cls)
            
            N_way_mask.append(mask)
            proposal_masks.append(query_proposal_mask.view(B, T, *query_proposal_mask.shape[-2:]))
            motion_cls.append(q_motion_cls)
        mask = torch.stack(N_way_mask, dim=1)
        proposal_masks = torch.stack(proposal_masks, dim=1)
        motion_cls = torch.stack(motion_cls, dim=1)
        # motion_cls = (motion_cls, q_cls_vec, s_cls_vec) if require_cls else motion_cls
        motion_cls = (motion_cls, q_mask_feats, s_cls_vec) if require_cls else motion_cls
        
        return mask, proposal_masks, motion_cls
    

class MaskProposalGeneratorAdj(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.conv_f32 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_f16 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_f8 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
    
    def forward(self, q_f32, q_f16, q_f8):
        # q_f32: (B, c, h/32, w/32)
        # q_f16: (B, c, h/16, w/16)
        # q_f8: (B, c, h/8, w/8)
        # motion_p: (N, B, hidden_dim)
        # motion_pe: (N, B, hidden_dim)
        
        B = q_f16.shape[0]
        x = self.conv_f32(q_f32) + q_f32
        x = F.interpolate(
            x, q_f16.shape[-2:], mode='bilinear', align_corners=False
        )
        x = x + q_f16
        x = self.conv_f16(x) + q_f16
        x = F.interpolate(
            x, q_f8.shape[-2:], mode='bilinear', align_corners=False
        )
        x = x + q_f8
        x = self.conv_f8(x) + q_f8
        x = self.mask_conv(x)
        return x.squeeze(1)


class MaskProposalGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False
            ) for _ in range(2)
        ])
        
        # Feature fusion convs
        self.fusion_conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
    
    def forward(self, q_f32, q_f16, q_f8, motion_p, motion_pe):
        # q_f32: (B, c, h/32, w/32)
        # q_f16: (B, c, h/16, w/16)
        # q_f8: (B, c, h/8, w/8)
        # motion_p: (N, B, hidden_dim)
        # motion_pe: (N, B, hidden_dim)
        
        B = q_f16.shape[0]
        T = q_f16.shape[0] // motion_p.shape[1]
        
        # Reshape features for cross attention
        f16_flat = q_f16.flatten(-2).permute(2, 0, 1)  # (HW, B, C)
        f8_flat = q_f8.flatten(-2).permute(2, 0, 1)  # (HW, B, C)
        
        motion_p = motion_p.repeat(1, T, 1)
        motion_pe = motion_pe.repeat(1, T, 1)
                
        # Cross attention with motion prototype
        f16_enhanced = self.transformer_cross_attention_layers[0](
            f16_flat, motion_p,
            pos=motion_pe,
            query_pos=None
        )
        
        f8_enhanced = self.transformer_cross_attention_layers[1](
            f8_flat, motion_p,
            pos=motion_pe,
            query_pos=None
        )
        
        # Reshape back to spatial features
        f16_enhanced = f16_enhanced.permute(1, 2, 0).view(B, -1, *q_f16.shape[-2:])  # (B, C, H/16, W/16)
        f8_enhanced = f8_enhanced.permute(1, 2, 0).view(B, -1, *q_f8.shape[-2:])  # (B, C, H/8, W/8)
        
        # Progressive feature fusion
        f16_up = F.interpolate(f16_enhanced, size=q_f8.shape[-2:], mode='bilinear', align_corners=False)
        fused_8 = self.fusion_conv1(torch.cat([f8_enhanced, f16_up], dim=1))  # (B, C, H/8, W/8)
        
        # Predict mask
        mask = self.mask_conv(fused_8)  # (B, 1, H/8, W/8)
        
        return mask.squeeze(1)


class MotionProtoptyeNetwork(nn.Module):
    def __init__(self, hidden_dim, num_meta_motion_queries=1, num_q_former_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_meta_motion_queries = num_meta_motion_queries
        self.num_q_former_layers = num_q_former_layers
        
        # Feature mapping layers
        self.motion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.appear_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # q-former
        # self.motion_cls = nn.Linear(1, self.num_meta_motion_queries)
        self.meta_motion_queries = nn.Embedding(1 + self.num_meta_motion_queries, self.hidden_dim)
        self.query_pos_embed = nn.Embedding(1 + self.num_meta_motion_queries, self.hidden_dim)
        # self.query_pos_embed = self.get_sinusoid_encoding(torch.arange(1 + self.num_meta_motion_queries), self.hidden_dim)
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_1 = nn.ModuleList()
        self.transformer_cross_attention_layers_2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_q_former_layers):
            self.transformer_cross_attention_layers_1.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers_2.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_dim*2,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.meta_motion_queries.weight, std=0.02)
        nn.init.normal_(self.query_pos_embed.weight, std=0.02)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def get_sinusoid_encoding(self, time_pos, d_hid):
        """Sinusoidal position encoding table supporting single sequence"""
        # Input is time position tensor
        seq_len = time_pos.shape[0]
        # Create position index
        position = time_pos.float().unsqueeze(-1)  # [seq_len, 1]
        
        # Calculate divisor term
        div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float, device=self.device) * 
                           (-math.log(10000.0) / d_hid))  # [d_hid/2]
        
        # Initialize encoding table
        sinusoid_table = torch.zeros(seq_len, d_hid, device=self.device)
        
        # Calculate sine and cosine encoding
        sinusoid_table[..., 0::2] = torch.sin(position * div_term)  # Even dimensions use sine
        sinusoid_table[..., 1::2] = torch.cos(position * div_term)  # Odd dimensions use cosine
        
        return sinusoid_table
        
    def forward(self, features, motion_feats, k_shot, num_support_frames):
        # features: (B, K, T, feat_dim)
        # motion_features: (B, K, T-1, feat_dim)
        B, K, T, feat_dim = features.shape
        
        # 1. Extract motion and appearance features
        # Calculate motion features from differences between adjacent frames
        # motion_feats = features[..., 1:, :] - features[..., :-1, :]
        zero_motion = torch.zeros(B, K, 1, feat_dim, device=self.device)
        motion_feats = torch.cat([motion_feats, zero_motion], dim=2)
        
        # Map motion and appearance features to hidden_dim
        motion_feats = self.motion_proj(motion_feats)
        appear_feats = self.appear_proj(features)
        
        # 2. Add position encoding
        # Temporal position encoding
        time_pos = torch.arange(num_support_frames, device=self.device).repeat(k_shot)
        time_pos_embed = self.get_sinusoid_encoding(time_pos, self.hidden_dim)
        
        # Shot position encoding
        shot_pos = torch.arange(k_shot, device=self.device).repeat_interleave(num_support_frames)
        shot_pos_embed = self.get_sinusoid_encoding(shot_pos, self.hidden_dim)
        
        # Merge position encodings
        pos_embed = time_pos_embed + shot_pos_embed
        pos_embed = pos_embed.unsqueeze(1)
        pos_embed = pos_embed.repeat(1, B, 1)
        
        # 3. Initialize meta-motion queries
        motion_q = self.meta_motion_queries.weight.unsqueeze(1).repeat(1, B, 1)
        motion_pe = self.query_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        
        try:
            motion_feats = motion_feats.view(B, k_shot * num_support_frames, self.hidden_dim).permute(1, 0, 2)
            appear_feats = appear_feats.view(B, k_shot * num_support_frames, self.hidden_dim).permute(1, 0, 2)
        except Exception as e:
            print(e)
            print(motion_feats.shape)
            print(appear_feats.shape)
            raise e
        
        # 4. Extract prototype through q-former
        for i in range(self.num_q_former_layers):
            motion_p = self.transformer_cross_attention_layers_1[i](motion_q, motion_feats, pos=pos_embed, query_pos=motion_pe)
            motion_p = self.transformer_self_attention_layers[i](motion_p, tgt_mask=None, tgt_key_padding_mask=None, query_pos=motion_pe)
            motion_p = self.transformer_cross_attention_layers_2[i](motion_p, appear_feats, pos=pos_embed, query_pos=motion_pe)
            motion_p = self.transformer_ffn_layers[i](motion_p)
            
        return motion_p, motion_pe  # (num_queries, B, hidden_dim)
        

class PrototypeEnhancer(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.cls_head_1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.cls_head_2 = nn.Linear(hidden_dim // 4, hidden_dim // 16)
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_dim*2,
                    dropout=0.0,
                )
            )
            
    def forward(self, motion1_p, motion1_pe, motion2_p, motion2_pe, require_cls=False):
        cls_token_1 = self.cls_head_2(self.cls_head_1(motion1_p[0]))
        cls_token_2 = self.cls_head_2(self.cls_head_1(motion2_p[0]))
        norm1 = torch.norm(cls_token_1, p=2, dim=1)
        norm2 = torch.norm(cls_token_2, p=2, dim=1)
        cls_score = torch.sum(cls_token_1 * cls_token_2, dim=1) / (norm1 * norm2 + 1e-8) * 5
        for i in range(self.num_layers):
            motion1_p = self.transformer_cross_attention_layers[i](motion1_p, motion2_p, pos=motion2_pe, query_pos=motion1_pe)
            motion1_p = self.transformer_self_attention_layers[i](motion1_p, tgt_mask=None, tgt_key_padding_mask=None, query_pos=motion1_pe)
            motion1_p = self.transformer_ffn_layers[i](motion1_p)
        
        if not require_cls:
            return motion1_p[1:], motion1_pe[1:], cls_score.unsqueeze(1)
        else:
            return motion1_p[1:], motion1_pe[1:], cls_score.unsqueeze(1), motion1_p[1], motion2_p[1]
    
    
class MotionAwareDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False
            ) for _ in range(3)
        ])
        
        # Feature fusion convs
        self.fusion_conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
        
    def forward(self, enhance_q_motion_p, enhance_q_motion_pe, f16, f8, f4):
        # enhance_q_motion_p: (1, B, hidden_dim)
        # enhance_q_motion_pe: (1, B, hidden_dim) 
        # f16: (B, T, hidden_dim, H/16, W/16)
        # f8: (B, T, hidden_dim, H/8, W/8)
        # f4: (B, T, hidden_dim, H/4, W/4)
        
        B, T = f16.shape[0], f16.shape[1]
        # Reshape features for cross attention
        f16_flat = f16.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f8_flat = f8.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f4_flat = f4.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        
        # Expand motion prototype to match batch size
        enhance_q_motion_p = enhance_q_motion_p.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        enhance_q_motion_pe = enhance_q_motion_pe.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        
        # Cross attention with motion prototype
        f16_enhanced = self.transformer_cross_attention_layers[0](
            f16_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f8_enhanced = self.transformer_cross_attention_layers[1](
            f8_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f4_enhanced = self.transformer_cross_attention_layers[2](
            f4_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        # Reshape back to spatial features
        f16_enhanced = f16_enhanced.permute(1, 2, 0).view(B, T, -1, *f16.shape[-2:])  # (B, T, C, H/16, W/16)
        f8_enhanced = f8_enhanced.permute(1, 2, 0).view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        f4_enhanced = f4_enhanced.permute(1, 2, 0).view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        
        # Progressive feature fusion
        f16_up = F.interpolate(f16_enhanced.flatten(0,1), size=f8.shape[-2:], mode='bilinear', align_corners=False)
        # f16_up = f16_up.view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        fused_8 = self.fusion_conv1(torch.cat([f8_enhanced.flatten(0,1), f16_up], dim=1))  # (B, T, C, H/8, W/8)
        
        fused_8_up = F.interpolate(fused_8, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # fused_8_up = fused_8_up.view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        fused_4 = self.fusion_conv2(torch.cat([f4_enhanced.flatten(0,1), fused_8_up], dim=1))  # (B, T, C, H/4, W/4)
        
        # Predict mask
        mask = self.mask_conv(fused_4)  # (B*T, 1, H/4, W/4)
        mask = mask.view(B, T, *mask.shape[-2:])  # (B, T, H/4, W/4)
        
        return mask
    

class MotionAwareDecoderAdj(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False
            ) for _ in range(3)
        ])
        
        # Feature fusion convs
        self.fusion_conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)

        self.mask_conv16 = nn.Conv2d(hidden_dim+1, hidden_dim, 3, padding=1)
        self.mask_conv8 = nn.Conv2d(hidden_dim+1, hidden_dim, 3, padding=1)
        self.mask_conv4 = nn.Conv2d(hidden_dim+1, hidden_dim, 3, padding=1)
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
        
    def forward(self, enhance_q_motion_p, enhance_q_motion_pe, f16, f8, f4, prior_mask):
        # enhance_q_motion_p: (1, B, hidden_dim)
        # enhance_q_motion_pe: (1, B, hidden_dim) 
        # f16: (B, T, hidden_dim, H/16, W/16)
        # f8: (B, T, hidden_dim, H/8, W/8)
        # f4: (B, T, hidden_dim, H/4, W/4)
        
        B, T = f16.shape[0], f16.shape[1]

        if prior_mask.ndim < 4:
            prior_mask = prior_mask.unsqueeze(1)

        mask_16 = F.interpolate(prior_mask, size=f16.shape[-2:], mode='bilinear', align_corners=False)
        mask_8 = F.interpolate(prior_mask, size=f8.shape[-2:], mode='bilinear', align_corners=False)
        mask_4 = F.interpolate(prior_mask, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # Reshape features for cross attention
        f16_flat = f16.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f8_flat = f8.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f4_flat = f4.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        
        # Expand motion prototype to match batch size
        enhance_q_motion_p = enhance_q_motion_p.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        enhance_q_motion_pe = enhance_q_motion_pe.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        
        # Cross attention with motion prototype
        f16_enhanced = self.transformer_cross_attention_layers[0](
            f16_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f8_enhanced = self.transformer_cross_attention_layers[1](
            f8_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f4_enhanced = self.transformer_cross_attention_layers[2](
            f4_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        # Reshape back to spatial features
        f16_enhanced = f16_enhanced.permute(1, 2, 0).view(B, T, -1, *f16.shape[-2:])  # (B, T, C, H/16, W/16)
        f8_enhanced = f8_enhanced.permute(1, 2, 0).view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        f4_enhanced = f4_enhanced.permute(1, 2, 0).view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        
        # Progressive feature fusion
        f16_enhanced = self.mask_conv16(torch.cat([f16_enhanced.flatten(0,1), mask_16], dim=1))
        f16_up = F.interpolate(f16_enhanced, size=f8.shape[-2:], mode='bilinear', align_corners=False)
        # f16_up = f16_up.view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        f8_enhanced = self.mask_conv8(torch.cat([f8_enhanced.flatten(0,1), mask_8], dim=1))
        fused_8 = self.fusion_conv1(torch.cat([f8_enhanced, f16_up], dim=1))  # (B, T, C, H/8, W/8)
        
        fused_8_up = F.interpolate(fused_8, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # fused_8_up = fused_8_up.view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        f4_enhanced = self.mask_conv16(torch.cat([f4_enhanced.flatten(0,1), mask_4], dim=1))
        fused_4 = self.fusion_conv2(torch.cat([f4_enhanced, fused_8_up], dim=1))  # (B, T, C, H/4, W/4)
        
        # Predict mask
        mask = self.mask_conv(fused_4)  # (B*T, 1, H/4, W/4)
        mask = mask.view(B, T, *mask.shape[-2:])  # (B, T, H/4, W/4)
        
        return mask


def getPositionEncoding1d(seq_length, feat_dim):
    position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)  # (q_frames, 1)
    div_term = torch.exp(torch.arange(0, feat_dim, 2, dtype=torch.float) * -(math.log(10000.0) / feat_dim))  # (D/2,)
    tem_pe = torch.zeros(seq_length, feat_dim)  # (q_frames, hidden_dim)
    tem_pe[:, 0::2] = torch.sin(position * div_term)  # sin encoding with hidden_dim dimension
    tem_pe[:, 1::2] = torch.cos(position * div_term)  # cos encoding with hidden_dim dimension
    return tem_pe
