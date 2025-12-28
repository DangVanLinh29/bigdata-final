import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')

import math
import argparse
import json
import os
import time
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libs.config.DMA_config import OPTION as opt
from libs.utils.misc import set_random_seed
from libs.models.DMA import DMA
from libs.dataset.MOVE import MOVEDataset
from libs.dataset.transform import TrainTransform
from libs.models.DMA.loss import build_criterion


SNAPSHOT_DIR = opt.SNAPSHOT_DIR


def setup_ddp():
    """Initialize DDP environment"""
    os.environ["USE_LIBUV"] = "0"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
    return local_rank


def cleanup_ddp():
    """Cleanup DDP environment"""
    dist.destroy_process_group()


def get_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MOVE Training')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--total_episodes", type=int, default=100000, help="total number of episodes for training")
    parser.add_argument("--support_frames", type=int, default=5)
    parser.add_argument("--query_frames", type=int, default=5)
    parser.add_argument("--num_ways", type=int, default=2, help="number of ways (N) in N-way-K-shot setting")
    parser.add_argument("--num_shots", type=int, default=2, help="number of shots (K) in N-way-K-shot setting")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--save_interval", type=int, default=200, help="Save model every N episodes")
    parser.add_argument("--print_interval", type=int, default=10, help="Print progress every N episodes")
    parser.add_argument("--validate_interval", type=int, default=4000, help="Print progress every N episodes")
    parser.add_argument("--ce_loss_weight", type=float, default=1.0, help="Weight for cross entropy loss")
    parser.add_argument("--iou_loss_weight", type=float, default=1.0, help="Weight for IoU loss")
    parser.add_argument("--cls_loss_weight", type=float, default=1.0, help="Weight for motion classification loss")
    parser.add_argument("--orth_loss_weight", type=float, default=1.0, help="Weight for motion classification loss")
    parser.add_argument("--obj_cls_loss_weight", type=float, default=1.0, help="Weight for object classification loss")
    parser.add_argument("--motion_cls_loss_weight", type=float, default=1.0, help="Weight for motion classification loss")
    parser.add_argument("--setting", type=str, default="default", help="default or challenging")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--warmup_episodes", type=int, default=500, help="Number of warmup episodes")
    parser.add_argument("--loss_type", type=str, default="default", help="default or mask2former")
    parser.add_argument("--backbone", type=str, default="resnet50", help="resnet or videoswin")
    parser.add_argument("--motion_appear_orth", action="store_true", help="Enable motion-appearance orthogonality and category classification")
    return parser.parse_args()


def get_warmup_cosine_schedule_with_warmup(optimizer, warmup_episodes, total_episodes):
    """Create learning rate scheduler with warmup and cosine annealing"""
    def lr_lambda(episode):
        if episode < warmup_episodes:
            return float(episode) / float(max(1, warmup_episodes))
        progress = float(episode - warmup_episodes) / float(max(1, total_episodes - warmup_episodes))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train():
    """Main training function"""
    args = get_arguments()
    
    # Setup DDP
    local_rank = setup_ddp()
    
    # Set random seed for reproducibility
    set_random_seed(seed=1234, deterministic=True)
    
    # Create snapshot directory only on main process
    save_dir = os.path.join(
        args.snapshot_dir, 
        args.backbone, 
        args.setting, 
        f'{args.num_ways}-way-{args.num_shots}-shot', 
        f'group{args.group}'
    )
    
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'train_log.txt')
        
        # Save complete parameter configuration
        with open(log_file, 'a') as f:
            f.write('='*50 + '\n')
            f.write('Running parameters:\n') 
            f.write('='*50 + '\n')
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')
            f.write('='*50 + '\n')
            f.write('\nFull configuration:\n')
            f.write(json.dumps(vars(args), indent=4, separators=(',', ':')) + '\n')
            f.write('='*50 + '\n\n')
        
        # Initialize tensorboard writer
        writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # Build model
    model = DMA(
        n_way=args.num_ways, 
        k_shot=args.num_shots, 
        num_support_frames=args.support_frames, 
        num_query_frames=args.query_frames, 
        backbone=args.backbone,
        motion_appear_orth=args.motion_appear_orth
    )
    
    if local_rank == 0:
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Move model to GPU and wrap with DDP
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Setup optimizer and gradient scaler for mixed precision training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_warmup_cosine_schedule_with_warmup(
        optimizer, args.warmup_episodes, args.total_episodes
    )
    scaler = GradScaler()

    # Try to load latest checkpoint if resume is True
    start_episode = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            if local_rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            best_loss = checkpoint['loss']
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Adjust learning rate scheduler
            for _ in range(start_episode):
                scheduler.step()

    # Setup datasets
    size = (241, 425)
    train_transform = TrainTransform(size)    
    train_dataset = MOVEDataset(
        train=True,
        group=args.group,
        support_frames=args.support_frames,
        query_frames=args.query_frames,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        transforms=train_transform,
        setting=args.setting,
        proposal_mask=True
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # Training loop
    if local_rank == 0:
        print(f'Start training from episode {start_episode + 1}')
        start_time = time.time()
    
    model.train()
    total_loss = 0
    moving_loss = 0  # For calculating moving average loss
    moving_mask_ce_loss = 0
    moving_proposal_ce_loss = 0
    moving_mask_iou_loss = 0
    moving_proposal_iou_loss = 0
    moving_motion_cls_loss = 0
    moving_gt_ce_loss = 0
    moving_gt_iou_loss = 0
    moving_gt_motion_cls_loss = 0
    moving_support_motion_cls_loss = 0
    moving_orth_loss = 0
    moving_orth_loss_gt = 0
    moving_obj_cls_loss = 0
    moving_motion_cls_loss_detailed = 0

    criterion = build_criterion(args.loss_type)
    
    for episode, (query_frames, query_masks, support_frames, support_masks, _, proposal_masks, support_object_categories, support_motion_categories) in enumerate(train_loader, start=start_episode):
        if episode >= args.total_episodes:
            break
            
        # Reshape data
        B, _, C, H, W = query_frames.shape
        query_frames = query_frames.view(args.batch_size, args.num_ways, -1, C, H, W)
        query_masks = query_masks.view(args.batch_size, args.num_ways, -1, H, W)
        proposal_masks = proposal_masks.view(args.batch_size, args.num_ways, -1, H, W)
        
        supp_F = args.support_frames
        K = args.num_shots
        support_frames = support_frames.view(args.batch_size, args.num_ways, K, supp_F, C, H, W)
        support_masks = support_masks.view(args.batch_size, args.num_ways, K, supp_F, H, W)
        
        # Move data to GPU
        query_frames = query_frames.cuda(local_rank, non_blocking=True)
        query_masks = query_masks.cuda(local_rank, non_blocking=True)
        proposal_masks = proposal_masks.cuda(local_rank, non_blocking=True)
        support_frames = support_frames.cuda(local_rank, non_blocking=True)
        support_masks = support_masks.cuda(local_rank, non_blocking=True)
        
        # Reshape category labels
        # Convert lists to tensors if needed
        if isinstance(support_object_categories, list):
            support_object_categories = torch.stack(support_object_categories, dim=0)
        if isinstance(support_motion_categories, list):
            support_motion_categories = torch.stack(support_motion_categories, dim=0)
            
        support_object_categories = support_object_categories.view(args.batch_size, args.num_ways, args.num_shots)
        support_motion_categories = support_motion_categories.view(args.batch_size, args.num_ways, args.num_shots)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            if args.motion_appear_orth:
                output, output_cls_motion, gt_otuput, gt_cls_motion, output_proposal_masks, support_output_motion_cls, support_gt_motion_cls, q_motion, q_motion_gt, support_category_logits, support_motion_logits, orthogonal_loss\
                      = model(query_frames[:, 0], support_frames, support_masks, query_masks)  # B, N_way, T, H, W
            else:
                output, output_cls_motion, gt_otuput, gt_cls_motion, output_proposal_masks, support_output_motion_cls, support_gt_motion_cls, q_motion, q_motion_gt\
                      = model(query_frames[:, 0], support_frames, support_masks, query_masks)  # B, N_way, T, H, W
                support_category_logits = None
                support_motion_logits = None
                orthogonal_loss = torch.tensor(0.0, device=output.device)
            if args.loss_type == 'default':
                output = F.interpolate(output.view(-1, 1, *output.shape[-2:]), size=(241, 425), mode='bilinear', align_corners=False).sigmoid()
                output = output.view(B, args.num_ways, -1, 1, *size)  # Reshape back to [B, N_way, T, 1, 241, 425]
                output = output[:, :, :, 0, :, :]
                
                output_proposal_masks = F.interpolate(output_proposal_masks.view(-1, 1, *output_proposal_masks.shape[-2:]), size=(241, 425), mode='bilinear', align_corners=False).sigmoid()
                output_proposal_masks = output_proposal_masks.view(B, args.num_ways, -1, 1, *size)
                output_proposal_masks = output_proposal_masks[:, :, :, 0, :, :]
                
                gt_output_masks = F.interpolate(gt_otuput.view(-1, 1, *gt_otuput.shape[-2:]), size=(241, 425), mode='bilinear', align_corners=False).sigmoid()
                gt_output_masks = gt_output_masks.view(B, args.num_ways, -1, 1, *size)
                gt_output_masks = gt_output_masks[:, :, :, 0, :, :]
    
            proposal_ce_loss, proposal_iou_loss = criterion(output_proposal_masks.flatten(1, 2), proposal_masks.flatten(1, 2))
            
            # cls_motion: B * N_way * 1: This is a logits used to indicate whether mask exists
            # Calculate motion classification loss
            cls_motion = output_cls_motion.view(B, args.num_ways)  # Reshape to (B, N_way)
            # Check if there are foreground pixels in query masks
            cls_target = (query_masks.sum(dim=(2,3,4)) > 0).float()  # B, N_way
            pos_weight = torch.ones_like(cls_target) * 3.0  # Weight for positive examples
            motion_cls_loss = F.binary_cross_entropy_with_logits(cls_motion, cls_target, pos_weight=pos_weight)

            orth_loss = proto_dist_loss(q_motion.permute(1, 0, 2))
            orth_loss_gt = proto_dist_loss(q_motion_gt.permute(1, 0, 2))
            
            # Only calculate loss for objects with foreground masks
            valid_mask = cls_target > 0
            
            # # Flatten and filter outputs and masks
            output_flat = output.flatten(0, 1)[valid_mask.flatten()]  # (valid_samples, H, W) 
            query_masks_flat = query_masks.flatten(0, 1)[valid_mask.flatten()]  # (valid_samples, H, W)
            
            if output_flat.numel() > 0:  # Only compute loss if there are valid samples
                mask_ce_loss, mask_iou_loss = criterion(output_flat, query_masks_flat)
            else:
                mask_ce_loss = torch.tensor(0.0, device=output.device)
                mask_iou_loss = torch.tensor(0.0, device=output.device)
            # mask_ce_loss, mask_iou_loss = criterion(output.flatten(1, 2), query_masks.flatten(1, 2))
            
            # Add motion classification loss to total loss
            
            # gt part
            # gt_output_flat = gt_output_masks.flatten(0, 1)[valid_mask.flatten()]
            # if gt_output_flat.numel() > 0:  # Only compute loss if there are valid samples
            #     gt_ce_loss, gt_iou_loss = criterion(gt_output_flat, query_masks_flat)
            # else:
            #     gt_ce_loss = torch.tensor(0.0, device=output.device)
            #     gt_iou_loss = torch.tensor(0.0, device=output.device)
            gt_ce_loss, gt_iou_loss = criterion(gt_output_masks.flatten(1, 2), query_masks.flatten(1, 2))
            # Apply higher penalty for false positives (predicting 1 when it should be 0)
            # pos_weight = torch.ones_like(cls_target) * 2.0  # Weight for positive examples
            gt_motion_cls_loss = F.binary_cross_entropy_with_logits(
                gt_cls_motion[..., 0], 
                cls_target, 
                pos_weight=pos_weight
            )
            support_motion_cls_loss = F.binary_cross_entropy_with_logits(support_output_motion_cls[..., 0], support_gt_motion_cls)
            
            # Category classification losses
            obj_cls_loss = torch.tensor(0.0, device=output.device)
            motion_cls_loss_detailed = torch.tensor(0.0, device=output.device)
            
            # Only apply motion_appear_orth losses for the first 1000 episodes
            if args.motion_appear_orth and support_category_logits is not None and episode < 1000:
                # Move category labels to GPU
                support_object_categories = support_object_categories.cuda(local_rank, non_blocking=True)
                support_motion_categories = support_motion_categories.cuda(local_rank, non_blocking=True)
                
                # Check if we have valid category labels (not all zeros)
                if support_object_categories.numel() > 0 and support_motion_categories.numel() > 0:
                    # Object classification loss using real labels
                    # support_category_logits: [B, N_way, 88], support_object_categories: [B, N_way, K]
                    obj_cls_loss = F.cross_entropy(support_category_logits.view(-1, 88), support_object_categories.view(-1).long())
                    
                    # Motion classification loss using real labels
                    # support_motion_logits: [B, N_way, 224], support_motion_categories: [B, N_way, K]
                    motion_cls_loss_detailed = F.cross_entropy(support_motion_logits.view(-1, 224), support_motion_categories.view(-1).long())
                else:
                    # Fallback to placeholder if no valid labels
                    obj_cls_loss = torch.tensor(0.0, device=output.device)
                    motion_cls_loss_detailed = torch.tensor(0.0, device=output.device)
            
            # Use smaller weights for motion_appear_orth losses
            motion_appear_weight = 0.1 if episode < 1000 else 0.0  # Small weight and only for first 1000 episodes
            
            total_loss = args.ce_loss_weight * (mask_ce_loss + proposal_ce_loss + gt_ce_loss) + args.iou_loss_weight * (mask_iou_loss + proposal_iou_loss + gt_iou_loss) \
                  + args.cls_loss_weight * (motion_cls_loss + gt_motion_cls_loss) + args.orth_loss_weight * (orth_loss + orth_loss_gt) \
                  + motion_appear_weight * (orthogonal_loss + args.obj_cls_loss_weight * obj_cls_loss + args.motion_cls_loss_weight * motion_cls_loss_detailed)
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        
        # Update parameters with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update loss statistics
        moving_mask_ce_loss = 0.9 * moving_mask_ce_loss + 0.1 * mask_ce_loss.item() if episode > 0 else mask_ce_loss.item()
        moving_proposal_ce_loss = 0.9 * moving_proposal_ce_loss + 0.1 * proposal_ce_loss.item() if episode > 0 else proposal_ce_loss.item()
        moving_mask_iou_loss = 0.9 * moving_mask_iou_loss + 0.1 * mask_iou_loss.item() if episode > 0 else mask_iou_loss.item()
        moving_proposal_iou_loss = 0.9 * moving_proposal_iou_loss + 0.1 * proposal_iou_loss.item() if episode > 0 else proposal_iou_loss.item()
        moving_motion_cls_loss = 0.9 * moving_motion_cls_loss + 0.1 * motion_cls_loss.item() if episode > 0 else motion_cls_loss.item()
        moving_gt_ce_loss = 0.9 * moving_gt_ce_loss + 0.1 * gt_ce_loss.item() if episode > 0 else gt_ce_loss.item()
        moving_gt_iou_loss = 0.9 * moving_gt_iou_loss + 0.1 * gt_iou_loss.item() if episode > 0 else gt_iou_loss.item()
        moving_gt_motion_cls_loss = 0.9 * moving_gt_motion_cls_loss + 0.1 * gt_motion_cls_loss.item() if episode > 0 else gt_motion_cls_loss.item()
        moving_support_motion_cls_loss = 0.9 * moving_support_motion_cls_loss + 0.1 * support_motion_cls_loss.item() if episode > 0 else support_motion_cls_loss.item()
        moving_orth_loss = 0.9 * moving_orth_loss + 0.1 * orth_loss.item() if episode > 0 else orth_loss.item()
        moving_orth_loss_gt = 0.9 * moving_orth_loss_gt + 0.1 * orth_loss_gt.item() if episode > 0 else orth_loss_gt.item()
        moving_obj_cls_loss = 0.9 * moving_obj_cls_loss + 0.1 * obj_cls_loss.item() if episode > 0 else obj_cls_loss.item()
        moving_motion_cls_loss_detailed = 0.9 * moving_motion_cls_loss_detailed + 0.1 * motion_cls_loss_detailed.item() if episode > 0 else motion_cls_loss_detailed.item()
        moving_loss = 0.9 * moving_loss + 0.1 * total_loss.item() if episode > 0 else total_loss.item()
        
        # Log to tensorboard
        if local_rank == 0:
            writer.add_scalar('Loss/Mask_CE', moving_mask_ce_loss, episode)
            writer.add_scalar('Loss/Proposal_CE', moving_proposal_ce_loss, episode)
            writer.add_scalar('Loss/Mask_IoU', moving_mask_iou_loss, episode)
            writer.add_scalar('Loss/Proposal_IoU', moving_proposal_iou_loss, episode)
            writer.add_scalar('Loss/Motion_Cls', moving_motion_cls_loss, episode)
            writer.add_scalar('Loss/GT_CE', moving_gt_ce_loss, episode)
            writer.add_scalar('Loss/GT_IoU', moving_gt_iou_loss, episode)
            writer.add_scalar('Loss/GT_Motion_Cls', moving_gt_motion_cls_loss, episode)
            writer.add_scalar('Loss/Support_Motion_Cls', moving_support_motion_cls_loss, episode)
            writer.add_scalar('Loss/Orthogonality', moving_orth_loss, episode)
            writer.add_scalar('Loss/Orthogonality_gt', moving_orth_loss_gt, episode)
            writer.add_scalar('Loss/Object_Cls', moving_obj_cls_loss, episode)
            writer.add_scalar('Loss/Motion_Cls_Detailed', moving_motion_cls_loss_detailed, episode)
            writer.add_scalar('Loss/Total', moving_loss, episode)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], episode)
            writer.add_scalar('Motion_Appear_Active', 1.0 if episode < 1000 and args.motion_appear_orth else 0.0, episode)
        
        # Print progress and write to log file
        if local_rank == 0 and (episode + 1) % args.print_interval == 0:
            total_loss = 0
            
            time_spent = time.time() - start_time
            time_per_episode = time_spent / (episode + 1 - start_episode)
            remaining_episodes = args.total_episodes - episode - 1
            eta_seconds = remaining_episodes * time_per_episode
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            log_message = (f'Episode [{episode+1}/{args.total_episodes}], '
                            f'Mask CE Loss: {moving_mask_ce_loss:.4f}, '
                            f'Proposal CE Loss: {moving_proposal_ce_loss:.4f}, '
                            f'Mask IoU Loss: {moving_mask_iou_loss:.4f}, '
                            f'Proposal IoU Loss: {moving_proposal_iou_loss:.4f}, '
                            f'Motion Cls Loss: {moving_motion_cls_loss:.4f}, '
                            f'GT CE Loss: {moving_gt_ce_loss:.4f}, '
                            f'GT IoU Loss: {moving_gt_iou_loss:.4f}, '
                            f'GT Motion Cls Loss: {moving_gt_motion_cls_loss:.4f}, '
                            f'Support Motion Cls Loss: {moving_support_motion_cls_loss:.4f}, '
                            f'Object Cls Loss: {moving_obj_cls_loss:.4f}, '
                            f'Motion Cls Detailed: {moving_motion_cls_loss_detailed:.4f}, '
                            f'Motion_Appear_Active: {"Yes" if episode < 1000 and args.motion_appear_orth else "No"}, '
                            f'Total Loss: {moving_loss:.4f}, '
                            f'LR: {scheduler.get_last_lr()[0]:.6f}, '
                            f'ETA: {eta}')
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        
        # Save checkpoint
        if local_rank == 0 and (episode + 1) % args.save_interval == 0:
            # Save latest checkpoint
            latest_checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
            torch.save({
                'episode': episode + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scaler_state_dict': scaler.state_dict(),
                'loss': moving_loss,
                'args': args,
            }, latest_checkpoint_path)
            print(f"Saved latest checkpoint at episode {episode+1}")

            # Save best checkpoint if current loss is better
            if moving_loss < best_loss:
                best_loss = moving_loss
                best_checkpoint_path = os.path.join(save_dir, f'DAN_MoVe_episode{episode+1}_loss{moving_loss:.4f}.pth')
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': best_loss,
                    'args': args,
                }, best_checkpoint_path)
                print(f"Saved best checkpoint at episode {episode+1} with loss {moving_loss:.4f}")
        
        # Synchronize processes
        dist.barrier()
    
    # Close tensorboard writer
    if local_rank == 0:
        writer.close()
    
    # Cleanup
    cleanup_ddp()


def proto_dist_loss(proto_token):
    device = proto_token.device
    proto_num = proto_token.shape[1]
    negative_ = (torch.ones(proto_num) - torch.eye(proto_num)).to(device)
    cos_sim = batch_cos_sim(proto_token, proto_token)
    cos_dist = cos_sim * negative_
    # cos_dist = negative_ - cos_sim
    loss = cos_dist.exp().mean(dim=2).log().mean()
    return loss


def batch_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # calculate cosine similarity between a and b
    # a: [batch,num_a,channel]
    # b: [batch,num_b,channel]
    # return: [batch,num_a,num_b]
    assert a.shape[0] == b.shape[0], 'batch size of a and b must be equal'
    assert a.shape[2] == b.shape[2], 'channel of a and b must be equal'
    cos_esp = 1e-8
    a_norm = a.norm(dim=2, keepdim=True)
    b_norm = b.norm(dim=2, keepdim=True)
    cos_sim = torch.bmm(a, b.permute(0, 2, 1))
    cos_sim = cos_sim / (torch.bmm(a_norm, b_norm.permute(0, 2, 1)) + cos_esp)
    return cos_sim


if __name__ == '__main__':
    train()
