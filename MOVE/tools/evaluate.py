import sys
sys.path.append('./')

import pickle
import numpy as np
from pycocotools import mask as mask_util

from libs.utils.track_progress_rich import track_progress_rich
from libs.utils.davis_JF import db_eval_boundary, db_eval_iou


def setup_metrics(num_classes):
    """Initialize evaluation metrics"""
    # For non-empty masks
    f_list = [0] * num_classes
    j_list = [0] * num_classes
    n_list = [0] * num_classes
    j_fg_list = [0] * num_classes  # Foreground IoU
    j_bg_list = [0] * num_classes  # Background IoU
    
    # For calculating t-acc and n-acc
    t_acc_correct = 0  # Number of episodes correctly predicted as having foreground
    t_acc_total = 0    # Total number of episodes containing foreground
    n_acc_correct = 0  # Number of episodes correctly predicted as having no foreground
    n_acc_total = 0    # Total number of episodes without foreground

    return f_list, j_list, n_list, j_fg_list, j_bg_list, t_acc_correct, t_acc_total, n_acc_correct, n_acc_total


def process_episode(episode, idx):
    """Process a single episode"""
    pred_masks = episode['predictions']
    gt_masks = episode['query_masks']
    categories = episode['categories']
    class_indexes = episode['class_list'] + ['background']
    num_classes = len(class_indexes)
    
    f_list, j_list, n_list, j_fg_list, j_bg_list, t_acc_correct, t_acc_total, n_acc_correct, n_acc_total = setup_metrics(num_classes)

    # Process each batch
    for batch_idx in range(len(pred_masks)):
        pred_batch = pred_masks[batch_idx]
        gt_batch = gt_masks[batch_idx]
        gt_category = categories
        
        # Convert one-hot format predictions and ground truth to T*H*W format, 0 for background, 1~N for each class
        T = len(pred_batch[0])  # Get time dimension
        H = mask_util.decode(pred_batch[0][0]).shape[0]  # Get height
        W = mask_util.decode(pred_batch[0][0]).shape[1]  # Get width
        
        # Initialize converted prediction and ground truth arrays
        converted_pred = np.zeros((T, H, W), dtype=np.uint8)
        converted_gt = np.zeros((T, H, W), dtype=np.uint8)
        
        # Iterate through each way
        for n in range(len(pred_batch)):
            pred_way = pred_batch[n]
            gt_way = gt_batch[n]
            
            # Iterate through each time step
            for t in range(T):
                # Decode current frame's prediction and ground truth
                pred_mask = mask_util.decode(pred_way[t])
                gt_mask = mask_util.decode(gt_way[t])
                
                # Assign foreground class (n+1) to corresponding positions
                converted_pred[t][pred_mask > 0] = n
                converted_gt[t][gt_mask > 0] = n
        
        pred_batch = converted_pred
        gt_batch = converted_gt
        
        # Get all non-background classes in current batch
        unique_classes = np.unique(gt_batch)
        # Remove background class (0)
        unique_classes = unique_classes[unique_classes != 0]
        
        # Check if ground truth contains foreground
        has_foreground = len(unique_classes) > 0
        
        if not has_foreground:
            # If ground truth contains no foreground
            n_acc_total += 1
            if np.all(pred_batch == 0):  # If prediction is also all background
                n_acc_correct += 1
        else:
            # If ground truth contains foreground
            t_acc_total += 1
            if np.any(pred_batch > 0):  # If prediction contains foreground
                t_acc_correct += 1
        
        # Create binary masks - all foreground as 1, background as 0
        pred_binary_fg = (pred_batch > 0).astype(np.uint8)
        gt_binary_fg = (gt_batch > 0).astype(np.uint8)
        
        # Create binary masks - all background as 1, foreground as 0
        pred_binary_bg = (pred_batch == 0).astype(np.uint8)
        gt_binary_bg = (gt_batch == 0).astype(np.uint8)
        
        # Calculate foreground IoU
        j_fg = db_eval_iou(gt_binary_fg, pred_binary_fg)
        if isinstance(j_fg, np.ndarray):
            j_fg = np.mean(j_fg)
            
        # Calculate background IoU
        j_bg = db_eval_iou(gt_binary_bg, pred_binary_bg)
        if isinstance(j_bg, np.ndarray):
            j_bg = np.mean(j_bg)
        
        for i in unique_classes:
            cate_name = gt_category[i-1][0]
            
            # Create binary masks for current class
            pred_binary = np.zeros_like(pred_batch, dtype=bool)
            gt_binary = np.zeros_like(gt_batch, dtype=bool)
            
            pred_binary[pred_batch == i] = True
            gt_binary[gt_batch == i] = True
            
            pred_binary = pred_binary.astype(np.uint8)
            gt_binary = gt_binary.astype(np.uint8)
            
            # Calculate J measure (IoU)
            j = []
            for t in range(len(gt_binary)):
                j_t = db_eval_iou(gt_binary[t], pred_binary[t])
                j.append(j_t)
            j = np.array(j)
            
            # Calculate F measure (boundary)
            f = []
            for t in range(len(gt_binary)):
                f_t = db_eval_boundary(gt_binary[t], pred_binary[t])
                f.append(f_t)
            f = np.array(f)
            
            # If multi-frame sequence, take average
            if isinstance(j, np.ndarray):
                j = np.mean(j)
            if isinstance(f, np.ndarray):
                f = np.mean(f)
                
            # Update corresponding class metric lists
            class_idx = class_indexes.index(cate_name)
            j_list[class_idx] += j
            f_list[class_idx] += f
            j_fg_list[class_idx] += j_fg
            j_bg_list[class_idx] += j_bg
            n_list[class_idx] += 1

    return {
        'j_list': j_list,
        'f_list': f_list,
        'j_fg_list': j_fg_list,
        'j_bg_list': j_bg_list,
        'n_list': n_list,
        'class_indexes': class_indexes,
        't_acc_correct': t_acc_correct,
        't_acc_total': t_acc_total,
        'n_acc_correct': n_acc_correct,
        'n_acc_total': n_acc_total
    }


def evaluate_move_results(results_file):
    """Evaluate MoVe inference results."""
    # Load results
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
        
    # Flatten results from all processes
    results = []
    for process_results in all_results:
        if process_results is not None:
            results.extend(process_results)

    # Process episodes in parallel
    episode_results = track_progress_rich(
        process_episode,
        results,
        nproc=32,
        description='Processing episodes'
    )

    # Get number of classes from first result
    num_classes = len(episode_results[0]['class_indexes'])
    class_indexes = episode_results[0]['class_indexes']
    
    # Initialize metrics
    f_list, j_list, n_list, j_fg_list, j_bg_list, t_acc_correct, t_acc_total, n_acc_correct, n_acc_total = setup_metrics(num_classes)

    # Combine results
    for result in episode_results:
        for i in range(num_classes):
            j_list[i] += result['j_list'][i]
            f_list[i] += result['f_list'][i]
            j_fg_list[i] += result['j_fg_list'][i]
            j_bg_list[i] += result['j_bg_list'][i]
            n_list[i] += result['n_list'][i]
        t_acc_correct += result['t_acc_correct']
        t_acc_total += result['t_acc_total']
        n_acc_correct += result['n_acc_correct']
        n_acc_total += result['n_acc_total']

    # Calculate final metrics
    metrics = get_metrics(j_list, f_list, j_fg_list, j_bg_list, n_list, class_indexes, 
                         t_acc_correct, t_acc_total, n_acc_correct, n_acc_total)
    return metrics


def get_metrics(j_list, f_list, j_fg_list, j_bg_list, n_list, class_indexes,
                t_acc_correct, t_acc_total, n_acc_correct, n_acc_total):
    """Get all evaluation metrics"""
    # Get indices of categories with test samples
    valid_indices = [i for i in range(len(n_list)) if n_list[i] > 0]
    
    # Calculate means only for categories with samples
    mean_f = np.mean([f_list[i]/n_list[i] for i in valid_indices]) if valid_indices else None
    mean_j = np.mean([j_list[i]/n_list[i] for i in valid_indices]) if valid_indices else None
    mean_j_fg = np.mean([j_fg_list[i]/n_list[i] for i in valid_indices]) if valid_indices else None
    mean_j_bg = np.mean([j_bg_list[i]/n_list[i] for i in valid_indices]) if valid_indices else None
    
    # Calculate t-acc and n-acc
    t_acc = t_acc_correct / t_acc_total if t_acc_total > 0 else None
    n_acc = n_acc_correct / n_acc_total if n_acc_total > 0 else None
    
    return {
        'n_list': n_list,
        'mean_f': mean_f,
        'mean_j': mean_j,
        'mean_j_fg': mean_j_fg,
        'mean_j_bg': mean_j_bg,
        't_acc': t_acc,
        'n_acc': n_acc,
        'per_category_metrics': {
            category: {
                'f_score': f_list[i]/n_list[i] if n_list[i] > 0 else float('nan'),
                'j_score': j_list[i]/n_list[i] if n_list[i] > 0 else float('nan'),
                'j_fg_score': j_fg_list[i]/n_list[i] if n_list[i] > 0 else float('nan'),
                'j_bg_score': j_bg_list[i]/n_list[i] if n_list[i] > 0 else float('nan'),
                'j_list': j_list[i],
                'f_list': f_list[i],
                'j_fg_list': j_fg_list[i],
                'j_bg_list': j_bg_list[i],
                'n_list': n_list[i]
            }
            for i, category in enumerate(class_indexes)
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate MoVe results')
    parser.add_argument('results_file', type=str, default="/path/to/results.pkl", help='Path to results pickle file')
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_move_results(args.results_file)
    
    # Print results in table format
    print("\nPer-category Results:")
    print("-" * 110)
    print(f"{'Category':<40} {'J-score':>12} {'F-score':>12} {'J-fg':>12} {'J-bg':>12} {'J&F':>12}")
    print("-" * 110)
    
    for category, category_metrics in metrics['per_category_metrics'].items():
        j_score = category_metrics['j_score']
        f_score = category_metrics['f_score']
        j_fg_score = category_metrics['j_fg_score']
        j_bg_score = category_metrics['j_bg_score']
        jf_score = (j_score + f_score) / 2 if not (np.isnan(j_score) or np.isnan(f_score)) else float('nan')
        
        print(f"{category:<40} {j_score:>12.8f} {f_score:>12.8f} {j_fg_score:>12.8f} {j_bg_score:>12.8f} {jf_score:>12.8f}")

    # Print overall metrics
    mean_jf = (metrics['mean_j'] + metrics['mean_f']) / 2 if metrics['mean_j'] is not None and metrics['mean_f'] is not None else float('nan')
    print(f"{'Overall':<40} {metrics['mean_j']:>12.8f} {metrics['mean_f']:>12.8f} {metrics['mean_j_fg']:>12.8f} {metrics['mean_j_bg']:>12.8f} {mean_jf:>12.8f}")
    print("-" * 110)
    
    # Print t-acc and n-acc
    print(f"\nt-acc: {metrics['t_acc']:.8f}")
    print(f"n-acc: {metrics['n_acc']:.8f}" if metrics['n_acc'] is not None else "n-acc: None")
    
    # Save results to txt file
    results_file_path = args.results_file
    if results_file_path.endswith('.pkl'):
        txt_file_path = results_file_path.replace('.pkl', '_score.txt')
    else:
        txt_file_path = results_file_path + '_score.txt'
    
    with open(txt_file_path, 'w') as f:
        f.write("MoVe Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Per-category Results:\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Category':<40} {'J-score':>12} {'F-score':>12} {'J-fg':>12} {'J-bg':>12} {'J&F':>12}\n")
        f.write("-" * 110 + "\n")
        
        for category, category_metrics in metrics['per_category_metrics'].items():
            j_score = category_metrics['j_score']
            f_score = category_metrics['f_score']
            j_fg_score = category_metrics['j_fg_score']
            j_bg_score = category_metrics['j_bg_score']
            jf_score = (j_score + f_score) / 2 if not (np.isnan(j_score) or np.isnan(f_score)) else float('nan')
            
            f.write(f"{category:<40} {j_score:>12.8f} {f_score:>12.8f} {j_fg_score:>12.8f} {j_bg_score:>12.8f} {jf_score:>12.8f}\n")
        
        # Write overall metrics
        mean_jf = (metrics['mean_j'] + metrics['mean_f']) / 2 if metrics['mean_j'] is not None and metrics['mean_f'] is not None else float('nan')
        f.write(f"{'Overall':<40} {metrics['mean_j']:>12.8f} {metrics['mean_f']:>12.8f} {metrics['mean_j_fg']:>12.8f} {metrics['mean_j_bg']:>12.8f} {mean_jf:>12.8f}\n")
        f.write("-" * 110 + "\n\n")
        
        # Write t-acc and n-acc
        f.write(f"t-acc: {metrics['t_acc']:.8f}\n")
        f.write(f"n-acc: {metrics['n_acc']:.8f}\n" if metrics['n_acc'] is not None else "n-acc: None\n")
    
    print(f"\nResults saved to: {txt_file_path}")


if __name__ == '__main__':
    main()
