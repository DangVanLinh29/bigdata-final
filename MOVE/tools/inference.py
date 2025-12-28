import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('./')

import os
import time
import datetime
import argparse
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from libs.config.DMA_config import OPTION as opt
from libs.dataset.transform import TestTransform
from libs.dataset.MOVE import MOVEDataset
from libs.models.DMA import DMA


def setup_ddp():
    """Initialize DDP environment"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
    return local_rank


def cleanup_ddp():
    """Cleanup DDP environment"""
    dist.destroy_process_group()


def get_arguments():
    parser = argparse.ArgumentParser(description='Test DAN-MoVe with DDP')
    parser.add_argument("--group", type=int, nargs='+', default=1)
    parser.add_argument("--support_frames", type=int, default=5, help="number of frames per shot")
    parser.add_argument("--query_frames", type=int, default=5, help="number of query frames")
    parser.add_argument("--num_ways", type=int, default=1, help="number of ways (N) in N-way-K-shot setting")
    parser.add_argument("--num_shots", type=int, default=2, help="number of shots (K) in N-way-K-shot setting")
    parser.add_argument("--snapshot", type=str, required=True, help="path to the trained model")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of test episodes")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=None, help="directory to save inference results")
    parser.add_argument("--setting", type=str, default="default", help="default or challenging")
    parser.add_argument("--backbone", type=str, default="resnet50", help="default or challenging")
    parser.add_argument("--mask_thr", type=float, default=0.5, help="mask threshold")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing prediction files")
    return parser.parse_args()


def test():
    args = get_arguments()
    
    # Setup DDP and get local rank
    local_rank = setup_ddp()
    
    # Ensure all processes have same random seed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Setup Model
    model_support_frames = max(3, args.support_frames)
    model = DMA(n_way=args.num_ways, k_shot=args.num_shots, num_support_frames=model_support_frames, backbone=args.backbone).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Load trained model
    if local_rank == 0:
        print(f"Loading trained model from {args.snapshot}")
    checkpoint = torch.load(args.snapshot, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Setup Dataloader
    size = (241, 425)  # Define target size for rescaling
    test_transform = TestTransform(size)
    
    test_dataset = MOVEDataset(
        data_path=opt.root_path,
        train=False,
        group=args.group,
        support_frames=args.support_frames,
        query_frames=args.query_frames,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        transforms=test_transform,
        setting=args.setting
    )
    
    # Setup distributed sampler
    sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Ensure multiprocessing compatibility
    mp.set_start_method('spawn', force=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one episode at a time
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler
    )
    
    # Create save directory - use snapshot path if save_dir not provided
    save_dir = Path(args.snapshot).parent
    # group_name = args.group if isinstance(args.group, int) else sum(args.group)
    # save_dir = Path(args.save_dir) / f'{args.backbone}' / f'{args.num_ways}-way-{args.num_shots}-shot' / f'group{group_name}'
    # print(save_dir)

    if local_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if prediction file already exists
        mask_thr = args.mask_thr
        results_file = save_dir / f'inference_results_thr{mask_thr}.pkl'
        if results_file.exists() and not args.overwrite:
            print(f"Prediction file {results_file} already exists. Use --overwrite to overwrite.")
            cleanup_ddp()
            print("\nStarting evaluation...")
            os.system("python tools/evaluate.py " + str(results_file))
            return
        
        print("Start testing...")
        start_time = time.time()
    
    results = []
    total_correct = 0
    total_samples = 0
    total_positive_correct = 0
    total_positive_samples = 0
    total_negative_correct = 0
    total_negative_samples = 0
    total_episodes = 0
    total_perfect_episodes = 0  # Count episodes where all ways are correctly classified
    
    # Add new metrics
    total_all_zero_episodes = 0  # Episodes where all ways are 0
    total_all_zero_correct = 0  # Correctly predicted all-zero episodes
    total_all_one_episodes = 0  # Episodes where all ways are 1
    total_all_one_correct = 0  # Correctly predicted all-one episodes
    total_mixed_episodes = 0  # Episodes with mix of 0s and 1s
    total_mixed_correct = 0  # Correctly predicted mixed episodes

    total_time = 0
    processed_frames = 0
    
    with torch.no_grad():
        for i, (query_frames, query_masks, support_frames, support_masks, video_ids, categories) in enumerate(test_loader):
            if i >= args.num_episodes:
                break
                
            try:
                # Move data to GPU
                query_frames = query_frames.to(device)
                query_masks = query_masks.to(device)
                support_frames = support_frames.to(device)
                support_masks = support_masks.to(device)
                
                # Get dimensions
                B, NT, C, H, W = query_frames.shape
                N = args.num_ways
                K = args.num_shots
                sup_F = args.support_frames
                T = NT // N
                
                # Reshape tensors for N-way processing
                query_frames = query_frames.view(B, N, T, C, H, W)
                query_masks = query_masks.view(B, N, T, H, W)
                support_frames = support_frames.view(B, N, K, sup_F, C, H, W)
                support_masks = support_masks.view(B, N, K, sup_F, H, W)

                padding_support_frame = support_frames[:, :, :, -1:].clone()
                padding_support_mask = support_masks[:, :, :, -1:].clone()
                while support_frames.shape[3] < 3:
                    support_frames = torch.cat([support_frames, padding_support_frame], dim=3)
                    support_masks = torch.cat([support_masks, padding_support_mask], dim=3)

                s_time = time.time()
                # Process in chunks to avoid OOM
                chunk_size = 50
                pred_maps = []
                pred_cls = []
                
                for start_idx in range(0, T, chunk_size):
                    end_idx = min(start_idx + chunk_size, T)
                    chunk_query = query_frames[:, 0, start_idx:end_idx]
                    chunk_mask = query_masks[:, :, start_idx:end_idx]

                    valid_indices = end_idx - start_idx
                    if end_idx - start_idx < 3:
                        padding_frame = chunk_query[:, -1:].clone()
                        padding_mask = chunk_mask[:, :, -1:].clone()
                        while chunk_query.shape[1] < 5:
                            chunk_query = torch.cat([chunk_query, padding_frame], dim=1)
                            chunk_mask = torch.cat([chunk_mask, padding_mask], dim=2)
                    
                    pred_map, _, motion_cls = model(chunk_query, support_frames, support_masks, chunk_mask)  # B, N_way, T, H, W
                    # codes below are used for oracle experiment
                    # pred_map, _, motion_cls = model(chunk_query, support_frames, support_masks, query_mask=chunk_mask, oracle_mask=True)
                    # pred_map = chunk_mask
                    
                    pred_map = pred_map[:, :, :valid_indices].contiguous()
                    
                    # Resize pred_map to match expected size [B, N_way, T, 1, 241, 425] using interpolation
                    pred_map = F.interpolate(pred_map.view(-1, 1, *pred_map.shape[-2:]), size=(241, 425), mode='bilinear', align_corners=False)
                    pred_map = pred_map.view(B, N, -1, 1, 241, 425)  # Reshape back to [B, N_way, T, 1, 241, 425]
                    
                    pred_maps.append(pred_map.sigmoid())
                    pred_cls.append(motion_cls[..., 0])
                
                cls_target = (query_masks.sum(dim=(2, 3, 4)) > 0).float()  # B, N_way
                pred_maps = torch.cat(pred_maps, dim=2)  # torch.Size([1, 2, 60, 1, 241, 425])
                pred_cls = torch.cat(pred_cls, dim=0)  # torch.Size([1, 2])
                pred_cls = torch.mean(pred_cls, dim=0, keepdim=True)  # torch.Size([B, N_way])

                e_time = time.time()
                total_time += e_time - s_time
                processed_frames += T
                fps = processed_frames / total_time

                # Calculate classification accuracy
                mask_thr = args.mask_thr
                pred_cls_binary = (pred_cls > 0).float()
                pred_cls_sigmoid = pred_cls.sigmoid()
                # codes below are used for oracle experiment
                # pred_cls = cls_target
                
                # Calculate overall accuracy
                correct = (pred_cls_binary == cls_target).sum().item()
                total = cls_target.numel()
                total_correct += correct
                total_samples += total
                
                # Calculate positive class accuracy (true positives)
                positive_mask = (cls_target == 1.0)
                positive_correct = ((pred_cls_binary == cls_target) & positive_mask).sum().item()
                positive_samples = positive_mask.sum().item()
                total_positive_correct += positive_correct
                total_positive_samples += positive_samples
                
                # Calculate negative class accuracy (true negatives)
                negative_mask = (cls_target == 0.0)
                negative_correct = ((pred_cls_binary == cls_target) & negative_mask).sum().item()
                negative_samples = negative_mask.sum().item()
                total_negative_correct += negative_correct
                total_negative_samples += negative_samples
                
                # Calculate perfect episode accuracy (all ways correctly classified)
                total_episodes += B
                for b in range(B):
                    if (pred_cls_binary[b] == cls_target[b]).all().item():
                        total_perfect_episodes += 1
                        
                    # Calculate new metrics
                    target = cls_target[b]
                    pred = pred_cls_sigmoid[b]
                    
                    # All zeros case
                    if (target == 0).all():
                        total_all_zero_episodes += 1
                        if (pred < mask_thr).all():
                            total_all_zero_correct += 1
                            
                    # All ones case        
                    elif (target == 1).all():
                        total_all_one_episodes += 1
                        if (pred > mask_thr).all():
                            total_all_one_correct += 1
                            
                    # Mixed case
                    else:
                        total_mixed_episodes += 1
                        # For mixed case, check if prediction with highest confidence matches ground truth
                        pred_above_thr = pred > mask_thr
                        if pred_above_thr.any():  # Only check if any prediction is above threshold
                            max_conf_idx = pred.argmax()
                            if target[max_conf_idx] == 1:  # Check if highest confidence prediction is correct
                                total_mixed_correct += 1
                
                # Modified merging logic: assign pred_cls values to mask regions above threshold for each way
                B, N, T, _, H, W = pred_maps.shape
                pred_cls_sigmoid = pred_cls.sigmoid()  # Ensure pred_cls is sigmoid activated
                
                # Create one-hot encoded masks
                one_hot_masks = torch.zeros(B, N+1, T, H, W, device=pred_maps.device)
                
                # Set background class (index 0)
                # If all way mask values are below threshold, it's background
                one_hot_masks[:, 0] = mask_thr
                
                # For each way, assign pred_cls values to regions above threshold
                for n in range(N):
                    # Get current way's mask
                    current_way_mask = pred_maps[:, n, :, 0]  # B x T x H x W
                    # Get current way's pred_cls value
                    current_way_cls = pred_cls_sigmoid[:, n, None, None, None]  # B x 1 x 1 x 1
                    
                    current_way_mask = current_way_mask * current_way_cls
                    one_hot_masks[:, n+1] = current_way_mask
                
                # Ensure each position has only one class as 1 (select class with maximum value)
                max_vals, max_indices = torch.max(one_hot_masks, dim=1, keepdim=True)
                one_hot_masks = torch.zeros_like(one_hot_masks).scatter_(1, max_indices, 1.0)
                
                pred_maps = one_hot_masks
                
                # query_masks torch.Size([1, 2, 60, 241, 425])
                # Convert query_masks to one-hot format
                one_hot_query_masks = torch.zeros(B, N+1, T, H, W, device=query_masks.device)
                
                # Set background class (index 0)
                background_mask = (query_masks.max(dim=1)[0] < 0.5)  # B x T x H x W
                one_hot_query_masks[:, 0] = background_mask
                
                # Set foreground classes (index 1 to N)
                for n in range(N):
                    one_hot_query_masks[:, n+1] = query_masks[:, n]
                
                # Ensure each position has only one class as 1
                max_vals, max_indices = torch.max(one_hot_query_masks, dim=1, keepdim=True)
                one_hot_query_masks = torch.zeros_like(one_hot_query_masks).scatter_(1, max_indices, 1.0)
                
                query_masks = one_hot_query_masks
                
                # Convert predictions and ground truth to binary masks
                binary_masks = (pred_maps.cpu().numpy()).astype(np.uint8)
                gt_masks = (query_masks.cpu().numpy()).astype(np.uint8)
                
                # Convert binary masks to RLE format
                rle_masks = []
                rle_gt_masks = []
                for b in range(binary_masks.shape[0]):
                    rle_per_batch = []
                    rle_gt_per_batch = []
                    for n in range(binary_masks.shape[1]):
                        rle_per_way = []
                        rle_gt_per_way = []
                        for t in range(binary_masks.shape[2]):
                            rle = mask_utils.encode(np.asfortranarray(binary_masks[b, n, t]))
                            rle_gt = mask_utils.encode(np.asfortranarray(gt_masks[b, n, t]))
                            rle_per_way.append(rle)
                            rle_gt_per_way.append(rle_gt)
                        rle_per_batch.append(rle_per_way)
                        rle_gt_per_batch.append(rle_gt_per_way)
                    rle_masks.append(rle_per_batch)
                    rle_gt_masks.append(rle_gt_per_batch)
                
                # Save results
                episode_result = {
                    'predictions': rle_masks,
                    'query_masks': rle_gt_masks,
                    'video_ids': video_ids,
                    'categories': categories,
                    'class_list': test_dataset.test_categories
                }
                results.append(episode_result)
                
                # Print progress and estimated remaining time
                if local_rank == 0 and (i + 1) % 10 == 0:
                    time_spent = time.time() - start_time
                    time_per_episode = time_spent / (i + 1)
                    remaining_episodes = args.num_episodes - i - 1
                    eta_seconds = remaining_episodes * time_per_episode
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
                    positive_accuracy = total_positive_correct / total_positive_samples * 100 if total_positive_samples > 0 else 0
                    negative_accuracy = total_negative_correct / total_negative_samples * 100 if total_negative_samples > 0 else 0
                    perfect_episode_accuracy = total_perfect_episodes / total_episodes * 100 if total_episodes > 0 else 0
                    
                    # Calculate new accuracy metrics
                    all_zero_accuracy = total_all_zero_correct / total_all_zero_episodes * 100 if total_all_zero_episodes > 0 else 0
                    all_one_accuracy = total_all_one_correct / total_all_one_episodes * 100 if total_all_one_episodes > 0 else 0
                    mixed_accuracy = total_mixed_correct / total_mixed_episodes * 100 if total_mixed_episodes > 0 else 0
                    
                    print(f'Episode [{i+1}/{args.num_episodes}], '
                          f'Frames per query: {T}, '
                          f'Overall Accuracy: {overall_accuracy:.2f}%, '
                          f'Positive Accuracy: {positive_accuracy:.2f}%, '
                          f'Negative Accuracy: {negative_accuracy:.2f}%, '
                          f'Perfect Episode Accuracy: {perfect_episode_accuracy:.2f}%, '
                          f'All-Zero Accuracy: {all_zero_accuracy:.2f}%, '
                          f'All-One Accuracy: {all_one_accuracy:.2f}%, '
                          f'Mixed-Case Accuracy: {mixed_accuracy:.2f}%, '
                          f'ETA: {eta}')
                    
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in episode {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

    # Gather results from all processes to rank 0
    if local_rank == 0:
        all_results = [results]
        for rank in range(1, dist.get_world_size()):
            rank_results = [None]
            dist.recv_object_list(rank_results, src=rank)
            all_results.extend(rank_results)
    else:
        dist.send_object_list([results], dst=0)
    
    # Save merged results from all processes
    if local_rank == 0:
        results_file = save_dir / f'inference_results_thr{mask_thr}.pkl'
            
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved to {results_file}")
    
    # Cleanup
    cleanup_ddp()
    
    # Run evaluation if this is the main process
    if local_rank == 0:
        print("\nStarting evaluation...")
        os.system("python tools/evaluate.py " + str(results_file))
        
        
if __name__ == '__main__':
    test()
