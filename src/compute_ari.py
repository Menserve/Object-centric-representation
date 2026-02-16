"""
ARI (Adjusted Rand Index) Computation for Object-Centric Models
================================================================

Computes FG-ARI (foreground-only) and Full-ARI metrics by comparing
predicted slot attention masks against ground-truth instance segmentations.

Usage:
    python src/compute_ari.py --checkpoint checkpoints/temp_scaling_tau05/dinov2_vits14/best_model.pt \
                              --data_dir data/movi_a_subset \
                              --backbone dinov2_vits14

    # With custom loss type (for channel_norm models):
    python src/compute_ari.py --checkpoint checkpoints/dinov1_channel_norm/dino_vits16/best_model.pt \
                              --data_dir data/movi_a_subset \
                              --backbone dino_vits16 \
                              --loss_type channel_norm
"""

import argparse
import os
import sys
import glob

import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset, collate_fn


def compute_ari(
    pred_masks: np.ndarray,
    gt_segmentation: np.ndarray,
    foreground_only: bool = True
) -> float:
    """
    Compute Adjusted Rand Index between predicted masks and GT segmentation.
    
    Args:
        pred_masks: (K, H, W) predicted slot attention masks (soft or hard)
        gt_segmentation: (H, W) ground-truth instance segmentation (integer labels)
        foreground_only: If True, exclude background (label 0) pixels
    
    Returns:
        ARI score (float in [-1, 1], higher is better)
    """
    H, W = gt_segmentation.shape
    
    # Convert soft masks to hard assignment: argmax over slots
    # pred_masks: (K, H, W) -> (H, W) with slot index per pixel
    pred_labels = pred_masks.argmax(axis=0).flatten()  # (H*W,)
    gt_labels = gt_segmentation.flatten()  # (H*W,)
    
    if foreground_only:
        # Exclude background pixels (label 0)
        fg_mask = gt_labels > 0
        if fg_mask.sum() == 0:
            return float('nan')
        pred_labels = pred_labels[fg_mask]
        gt_labels = gt_labels[fg_mask]
    
    return adjusted_rand_score(gt_labels, pred_labels)


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    backbone: str = 'dinov2_vits14',
    mask_temperature: float = 0.5,
    num_slots: int = 7,
    max_frames: int = 1,
    loss_type: str = 'mse',
    device: str = 'cuda'
) -> dict:
    """
    Load a checkpoint and compute ARI on the dataset.
    
    Returns:
        dict with fg_ari, full_ari, per_sample results
    """
    # Load model
    print(f"Loading model: {backbone} from {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Handle both raw state_dict and training checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Auto-detect num_slots from checkpoint
    if 'slot_attention.slots_mu' in state_dict:
        detected_slots = state_dict['slot_attention.slots_mu'].shape[1]
        if detected_slots != num_slots:
            print(f"  Auto-adjusting num_slots: {num_slots} -> {detected_slots}")
            num_slots = detected_slots
    
    model = SAViDinosaur(
        num_slots=num_slots,
        backbone=backbone,
        mask_temperature=mask_temperature
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset = MoviDataset(
        data_dir=data_dir,
        split='all',
        target_size=(224, 224),
        max_frames=max_frames
    )
    
    fg_aris = []
    full_aris = []
    per_sample = []
    
    print(f"Evaluating {len(dataset)} samples...")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            video = sample['video'].unsqueeze(0).to(device)  # (1, T, 3, H, W)
            gt_seg = sample['segmentation'].numpy()  # (T, H_orig, W_orig)
            
            # Forward pass
            result = model.forward_video(
                video,
                return_all_masks=True,
                loss_type=loss_type
            )
            
            # Get masks: (1, T, K, 1, H, W) -> (T, K, H, W)
            masks = result['all_masks'][0, :, :, 0].cpu().numpy()  # (T, K, H, W)
            
            T = masks.shape[0]
            sample_fg_aris = []
            sample_full_aris = []
            
            for t in range(T):
                pred_mask_t = masks[t]  # (K, H, W)
                gt_seg_t = gt_seg[t]  # (H_orig, W_orig)
                
                # Resize GT segmentation to match prediction size
                _, pred_h, pred_w = pred_mask_t.shape
                gt_h, gt_w = gt_seg_t.shape
                if gt_h != pred_h or gt_w != pred_w:
                    # Nearest-neighbor resize for segmentation labels
                    gt_seg_t_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(gt_seg_t).float().unsqueeze(0).unsqueeze(0),
                        size=(pred_h, pred_w),
                        mode='nearest'
                    ).squeeze().numpy().astype(int)
                else:
                    gt_seg_t_resized = gt_seg_t.astype(int)
                
                fg_ari = compute_ari(pred_mask_t, gt_seg_t_resized, foreground_only=True)
                full_ari = compute_ari(pred_mask_t, gt_seg_t_resized, foreground_only=False)
                
                if not np.isnan(fg_ari):
                    sample_fg_aris.append(fg_ari)
                if not np.isnan(full_ari):
                    sample_full_aris.append(full_ari)
            
            avg_fg = np.mean(sample_fg_aris) if sample_fg_aris else float('nan')
            avg_full = np.mean(sample_full_aris) if sample_full_aris else float('nan')
            
            fg_aris.append(avg_fg)
            full_aris.append(avg_full)
            
            sample_info = {
                'idx': idx,
                'fg_ari': avg_fg,
                'full_ari': avg_full,
                'has_metal': sample.get('has_metal', None)
            }
            per_sample.append(sample_info)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(dataset)}] FG-ARI: {avg_fg:.4f}, Full-ARI: {avg_full:.4f}")
    
    # Aggregate
    valid_fg = [x for x in fg_aris if not np.isnan(x)]
    valid_full = [x for x in full_aris if not np.isnan(x)]
    
    # Split by metal/non-metal
    metal_fg = [s['fg_ari'] for s in per_sample if s.get('has_metal') and not np.isnan(s['fg_ari'])]
    nonmetal_fg = [s['fg_ari'] for s in per_sample if not s.get('has_metal') and not np.isnan(s['fg_ari'])]
    
    results = {
        'fg_ari_mean': np.mean(valid_fg) if valid_fg else float('nan'),
        'fg_ari_std': np.std(valid_fg) if valid_fg else float('nan'),
        'full_ari_mean': np.mean(valid_full) if valid_full else float('nan'),
        'full_ari_std': np.std(valid_full) if valid_full else float('nan'),
        'metal_fg_ari': np.mean(metal_fg) if metal_fg else float('nan'),
        'nonmetal_fg_ari': np.mean(nonmetal_fg) if nonmetal_fg else float('nan'),
        'n_samples': len(dataset),
        'per_sample': per_sample
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute ARI for object-centric models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--data_dir', type=str, default='data/movi_a_subset',
                        help='Path to MOVi-A data directory')
    parser.add_argument('--backbone', type=str, default='dinov2_vits14',
                        choices=['dinov2_vits14', 'dino_vits16', 'clip_vitb16'],
                        help='Backbone architecture')
    parser.add_argument('--mask_temperature', type=float, default=0.5,
                        help='Mask temperature (τ)')
    parser.add_argument('--num_slots', type=int, default=7,
                        help='Number of slots')
    parser.add_argument('--max_frames', type=int, default=1,
                        help='Max frames per sample')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'cosine', 'channel_norm'],
                        help='Loss function type used during training')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        backbone=args.backbone,
        mask_temperature=args.mask_temperature,
        num_slots=args.num_slots,
        max_frames=args.max_frames,
        loss_type=args.loss_type,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("ARI Results")
    print("=" * 60)
    print(f"Backbone:      {args.backbone}")
    print(f"Loss type:     {args.loss_type}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Samples:       {results['n_samples']}")
    print("-" * 60)
    print(f"FG-ARI:        {results['fg_ari_mean']:.4f} ± {results['fg_ari_std']:.4f}")
    print(f"Full-ARI:      {results['full_ari_mean']:.4f} ± {results['full_ari_std']:.4f}")
    print(f"Metal FG-ARI:  {results['metal_fg_ari']:.4f}")
    print(f"Non-metal FG-ARI: {results['nonmetal_fg_ari']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
