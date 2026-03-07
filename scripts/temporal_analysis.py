#!/usr/bin/env python3
"""
時系列追跡分析（Temporal Tracking Analysis）
--------------------------------------------
SAVi-DINOSAURのforward_videoで24フレーム推論し、
スロットが同じ物体を追跡し続けるか、金属鏡面の特殊性を検証。

出力:
  - docs/paper/figures/temporal_*.png  (論文/ポスター用)
  - logs/temporal_results.csv          (定量データ)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from savi_dinosaur import SAViDinosaur

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'movi_a_v2'
FIG_DIR = ROOT / 'docs' / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_slot_consistency(all_masks):
    """
    フレーム間のスロット一貫性を計算。
    各フレームペア(t, t+1)でスロットマスクのIoUを計算し、
    ハンガリアンマッチングで最適対応を見つけ、平均IoUを返す。
    
    Args:
        all_masks: (T, K, 1, H, W)
    Returns:
        consistency_scores: list of float (T-1個)
        matched_indices: list of arrays (T-1個)
    """
    T, K = all_masks.shape[:2]
    consistency_scores = []
    matched_indices_list = []
    
    for t in range(T - 1):
        mask_t = all_masks[t, :, 0]    # (K, H, W)
        mask_t1 = all_masks[t+1, :, 0]  # (K, H, W)
        
        # Hard assignment
        assign_t = mask_t.argmax(dim=0)   # (H, W)
        assign_t1 = mask_t1.argmax(dim=0) # (H, W)
        
        # IoU matrix (K x K)
        iou_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                inter = ((assign_t == i) & (assign_t1 == j)).float().sum().item()
                union = ((assign_t == i) | (assign_t1 == j)).float().sum().item()
                iou_matrix[i, j] = inter / max(union, 1)
        
        # Hungarian matching (maximize IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_ious = [iou_matrix[r, c] for r, c in zip(row_ind, col_ind)]
        consistency_scores.append(np.mean(matched_ious))
        matched_indices_list.append(col_ind)
    
    return consistency_scores, matched_indices_list


def compute_slot_temporal_entropy(all_masks):
    """
    各スロットのマスクが時間方向でどれだけ安定か計算。
    スロットごとにマスクの重心位置の標準偏差を返す。
    
    Args:
        all_masks: (T, K, 1, H, W)
    Returns:
        stability: (K,) 重心の時間的変動（小さい=安定）
    """
    T, K, _, H, W = all_masks.shape
    
    # Per-slot center of mass over time
    centroids = np.zeros((T, K, 2))  # (T, K, [y, x])
    
    for t in range(T):
        for k in range(K):
            m = all_masks[t, k, 0].numpy()
            if m.sum() < 1e-6:
                centroids[t, k] = [H/2, W/2]
                continue
            yy, xx = np.meshgrid(range(H), range(W), indexing='ij')
            centroids[t, k, 0] = (m * yy).sum() / m.sum()
            centroids[t, k, 1] = (m * xx).sum() / m.sum()
    
    # Temporal stability = std of centroid over time
    stability = np.sqrt(np.sum(np.var(centroids, axis=0), axis=1))  # (K,)
    return stability, centroids


def analyze_scene_temporal(model, scene_id, image_size=224):
    """1シーンの時系列分析"""
    path = DATA_DIR / f'scene_{scene_id:04d}.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    video = data['video']  # (T, 3, H, W)
    seg = data['segmentation']  # (T, H, W)
    T = video.shape[0]
    
    # Resize video
    frames = []
    segs = []
    for t in range(T):
        frame = video[t]
        if frame.shape[0] not in [1, 3]:
            frame = frame.permute(2, 0, 1)
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        frame = F.interpolate(frame.unsqueeze(0), size=(image_size, image_size),
                              mode='bilinear', align_corners=False)
        frames.append(frame)
        
        seg_t = seg[t]
        seg_resized = F.interpolate(seg_t.unsqueeze(0).unsqueeze(0).float(),
                                     size=(image_size, image_size),
                                     mode='nearest').squeeze().long()
        segs.append(seg_resized)
    
    video_tensor = torch.cat(frames, dim=0).unsqueeze(0)  # (1, T, 3, H, W)
    
    with torch.no_grad():
        result = model.forward_video(video_tensor.to(DEVICE), return_all_masks=True)
    
    all_masks = result['all_masks'][0].cpu()  # (T, K, 1, H_feat, W_feat)
    all_slots = result['all_slots'][0].cpu()  # (T, K, D)
    
    # Upsample masks to image resolution
    T_out, K, _, Hf, Wf = all_masks.shape
    masks_up = []
    for t in range(T_out):
        m = F.interpolate(all_masks[t], size=(image_size, image_size),
                          mode='bilinear', align_corners=False)
        masks_up.append(m)
    masks_up = torch.stack(masks_up)  # (T, K, 1, H, W)
    
    # Slot consistency
    consistency, matched = compute_slot_consistency(masks_up)
    
    # Slot stability
    stability, centroids = compute_slot_temporal_entropy(masks_up)
    
    return {
        'scene_id': scene_id,
        'frames': torch.cat(frames, dim=0).cpu(),  # (T, 3, H, W)
        'segs': torch.stack(segs),  # (T, H, W)
        'masks': masks_up,   # (T, K, 1, H, W)
        'slots': all_slots,  # (T, K, D)
        'consistency': consistency,
        'mean_consistency': np.mean(consistency),
        'stability': stability,
        'centroids': centroids,
        'materials': data.get('materials', []),
        'has_metal': bool(data.get('has_metal', False)),
        'n_metal': data.get('metal_count', 0),
        'n_objects': data.get('num_instances', 0),
    }


def plot_temporal_tracking(result, max_frames=12):
    """1シーンの時系列トラッキング可視化"""
    scene_id = result['scene_id']
    frames = result['frames']  # (T, 3, H, W)
    masks = result['masks']    # (T, K, 1, H, W)
    segs = result['segs']      # (T, H, W)
    T = min(frames.shape[0], max_frames)
    K = masks.shape[1]
    
    # Select frames evenly
    frame_indices = np.linspace(0, frames.shape[0]-1, T, dtype=int)
    
    nrows = 3  # input, GT, argmax slot assignment
    fig, axes = plt.subplots(nrows, T, figsize=(2.5 * T, 2.5 * nrows))
    
    # Color map for slots
    slot_colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for col, t in enumerate(frame_indices):
        # Row 0: Input frame
        ax = axes[0, col]
        img_np = frames[t].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f't={t}', fontsize=11)
        ax.axis('off')
        
        # Row 1: GT segmentation
        ax = axes[1, col]
        ax.imshow(segs[t].numpy(), cmap='tab10', interpolation='nearest', vmin=0, vmax=10)
        ax.axis('off')
        
        # Row 2: Slot assignment (argmax)
        ax = axes[2, col]
        slot_assign = masks[t, :, 0].argmax(dim=0).numpy()  # (H, W)
        slot_rgb = slot_colors[slot_assign][:, :, :3]
        ax.imshow(slot_rgb)
        ax.axis('off')
    
    axes[0, 0].set_ylabel('Input', fontsize=13)
    axes[1, 0].set_ylabel('GT Seg', fontsize=13)
    axes[2, 0].set_ylabel('Slot Assign', fontsize=13)
    
    fig.suptitle(f"Scene {scene_id}: Temporal Tracking (Mean IoU={result['mean_consistency']:.3f}, {result['n_objects']} objects)",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    out = FIG_DIR / f'temporal_scene{scene_id:03d}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_consistency_overview(results_list):
    """全シーンの追跡一貫性の概要図"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    
    # 1. Consistency over time — best/worst showcase
    ax = axes[0]
    sorted_r = sorted(results_list, key=lambda x: x['mean_consistency'])
    showcase = sorted_r[:3] + sorted_r[-3:]  # worst 3 + best 3
    cmap = plt.cm.coolwarm
    for i, r in enumerate(showcase):
        c = cmap(i / (len(showcase) - 1))
        label = f"S{r['scene_id']:03d} (IoU={r['mean_consistency']:.2f})"
        ax.plot(range(len(r['consistency'])), r['consistency'],
                label=label, color=c, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Frame Pair (t, t+1)', fontsize=14)
    ax.set_ylabel('Mean Slot IoU', fontsize=14)
    ax.set_title('Tracking Consistency Over Time', fontsize=15)
    ax.legend(fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. Histogram of mean consistency
    ax = axes[1]
    all_cons = [r['mean_consistency'] for r in results_list]
    ax.hist(all_cons, bins=20, color='#42A5F5', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(all_cons), color='#E53935', linewidth=2, linestyle='--',
               label=f'Mean={np.mean(all_cons):.3f}')
    ax.set_xlabel('Mean Slot IoU Consistency', fontsize=14)
    ax.set_ylabel('# Scenes', fontsize=14)
    ax.set_title('Distribution of Tracking Quality', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Consistency vs #objects
    ax = axes[2]
    n_objs = [r['n_objects'] for r in results_list]
    ax.scatter(n_objs, all_cons, c='#FF9800', alpha=0.6, s=40, edgecolors='none')
    ax.set_xlabel('Number of Objects', fontsize=14)
    ax.set_ylabel('Mean Slot IoU Consistency', fontsize=14)
    ax.set_title('Tracking vs. Scene Complexity', fontsize=15)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Temporal Slot Tracking: Can SAVi-DINOSAUR Track Objects Across Frames?',
                 fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / 'temporal_consistency.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {out}")


def plot_slot_trajectories(result):
    """1シーンのスロット重心軌跡を可視化"""
    centroids = result['centroids']  # (T, K, 2)
    T, K, _ = centroids.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Background: first frame
    img_np = result['frames'][0].permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_np, 0, 1), alpha=0.4)
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        traj = centroids[:, k, :]  # (T, 2) [y, x]
        ax.plot(traj[:, 1], traj[:, 0], '-o', color=colors[k],
                markersize=3, linewidth=1.5, alpha=0.8,
                label=f'Slot {k} (σ={result["stability"][k]:.1f})')
        # Start marker
        ax.plot(traj[0, 1], traj[0, 0], 's', color=colors[k], markersize=8)
        # End marker
        ax.plot(traj[-1, 1], traj[-1, 0], '^', color=colors[k], markersize=8)
    
    ax.set_title(f'Scene {result["scene_id"]}: Slot Centroid Trajectories\n'
                 f'Mean IoU Consistency: {result["mean_consistency"]:.3f}',
                 fontsize=15)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.set_xlim(0, result['frames'].shape[3])
    ax.set_ylim(result['frames'].shape[2], 0)
    ax.axis('off')
    
    out = FIG_DIR / f'temporal_trajectory_s{result["scene_id"]:03d}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}")


if __name__ == '__main__':
    print("=== Temporal Tracking Analysis ===")
    print(f"Device: {DEVICE}")
    
    # Load best model: K=9 τ=0.5 224px
    ckpt_path = ROOT / 'checkpoints' / 'k_sweep' / 'dinov2_vits14_K9' / 'dinov2_vits14' / 'best_model.pt'
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SAViDinosaur(num_slots=9, backbone='dinov2_vits14', image_size=224)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE).eval()
    
    # Select diverse scenes: some metal, some rubber, varying object counts
    # Use first 40 scenes (20 metal, 20 mixed/rubber typically)
    n_analyze = 40
    all_results = []
    
    for sid in tqdm(range(n_analyze), desc="Temporal analysis"):
        path = DATA_DIR / f'scene_{sid:04d}.pt'
        if not path.exists():
            continue
        try:
            r = analyze_scene_temporal(model, sid)
            all_results.append(r)
        except Exception as e:
            print(f"  [WARN] Scene {sid}: {e}")
    
    print(f"\nAnalyzed {len(all_results)} scenes")
    
    # Summary
    all_cons = [r['mean_consistency'] for r in all_results]
    print(f"Mean consistency: {np.mean(all_cons):.4f} ± {np.std(all_cons):.4f}")
    print(f"Range: [{np.min(all_cons):.4f}, {np.max(all_cons):.4f}]")
    
    # Generate figures
    print("\n--- Generating temporal figures ---")
    
    # Best/worst tracking showcase
    sorted_all = sorted(all_results, key=lambda x: x['mean_consistency'])
    showcase_scenes = sorted_all[:2] + sorted_all[-2:]  # worst 2 + best 2
    
    for r in showcase_scenes:
        plot_temporal_tracking(r, max_frames=8)
        plot_slot_trajectories(r)
    
    # Overview figure
    plot_consistency_overview(all_results)
    
    # Save CSV
    import csv
    csv_path = ROOT / 'logs' / 'temporal_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'scene_id', 'mean_consistency', 'n_objects',
            'slot_stability_mean', 'slot_stability_std'])
        w.writeheader()
        for r in all_results:
            w.writerow({
                'scene_id': r['scene_id'],
                'mean_consistency': r['mean_consistency'],
                'n_objects': r['n_objects'],
                'slot_stability_mean': float(np.mean(r['stability'])),
                'slot_stability_std': float(np.std(r['stability'])),
            })
    print(f"✓ {csv_path}")
    
    print("\n✅ Temporal tracking analysis complete!")
