#!/usr/bin/env python3
"""
不確実性推定（Uncertainty Estimation）+ エラー分析
-------------------------------------------------
マスクのエントロピーマップからモデルが「迷っている」領域を特定し、
高エントロピーシーンの特徴をパターン化する。

出力:
  - docs/paper/figures/uncertainty_*.png  (論文/ポスター用)
  - logs/uncertainty_results.csv          (定量データ)
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

from savi_dinosaur import SAViDinosaur

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'movi_a_v2'
FIG_DIR = ROOT / 'docs' / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_mask_entropy(masks: torch.Tensor) -> torch.Tensor:
    """
    マスクのエントロピーを計算
    Args:
        masks: (K, 1, H, W) softmax後のマスク（各ピクセルでスロット間の確率分布）
    Returns:
        entropy: (H, W) エントロピーマップ
    """
    # masks は (K, 1, H, W) → (K, H, W)
    p = masks[:, 0, :, :]  # (K, H, W)
    # p はスロット方向に確率分布（softmaxで正規化済み）
    p = p.clamp(min=1e-8)
    entropy = -(p * p.log()).sum(dim=0)  # (H, W)
    # 正規化: 最大エントロピー = log(K)
    max_entropy = np.log(masks.shape[0])
    entropy_norm = entropy / max_entropy
    return entropy_norm


def analyze_all_scenes(model, image_size=224, n_scenes=300):
    """全シーンのエントロピー統計を計算"""
    results = []
    
    for sid in tqdm(range(n_scenes), desc="Computing entropy"):
        path = DATA_DIR / f'scene_{sid:04d}.pt'
        if not path.exists():
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        video = data['video']
        frame = video[0] if video.dim() == 4 else video
        if frame.shape[0] not in [1, 3]:
            frame = frame.permute(2, 0, 1)
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        frame = F.interpolate(frame.unsqueeze(0), size=(image_size, image_size),
                              mode='bilinear', align_corners=False)
        
        seg = data['segmentation']
        seg_frame = seg[0] if seg.dim() == 3 else seg
        seg_resized = F.interpolate(seg_frame.unsqueeze(0).unsqueeze(0).float(),
                                     size=(image_size, image_size),
                                     mode='nearest').squeeze().long()
        
        with torch.no_grad():
            img = frame.to(DEVICE)
            _, _, masks, slots = model.forward_image(img)
        
        # masks: (1, K, 1, H_feat, W_feat)
        masks_feat = masks[0]  # (K, 1, H_feat, W_feat)
        
        # Upsample masks to image resolution for visualization
        masks_up = F.interpolate(masks[0], size=(image_size, image_size),
                                 mode='bilinear', align_corners=False)
        
        # Entropy at feature resolution
        entropy = compute_mask_entropy(masks_feat)  # (H_feat, W_feat)
        entropy_up = compute_mask_entropy(masks_up)  # (H, W)
        
        # Statistics
        ent_np = entropy.cpu().numpy()
        ent_up_np = entropy_up.cpu().numpy()
        
        # Per-object entropy: entropy at each GT object's location
        gt_np = seg_resized.numpy()
        unique_objs = np.unique(gt_np)
        unique_objs = unique_objs[unique_objs > 0]  # Remove background
        
        materials = data.get('materials', [])
        n_metal = data.get('metal_count', 0)
        n_rubber = data.get('rubber_count', 0)
        
        results.append({
            'scene_id': sid,
            'mean_entropy': float(ent_up_np.mean()),
            'max_entropy': float(ent_up_np.max()),
            'std_entropy': float(ent_up_np.std()),
            'high_entropy_ratio': float((ent_up_np > 0.7).mean()),  # % of pixels with >70% max entropy
            'n_objects': len(unique_objs),
            'n_metal': n_metal if isinstance(n_metal, int) else int(n_metal),
            'n_rubber': n_rubber if isinstance(n_rubber, int) else int(n_rubber),
            'has_metal': bool(data.get('has_metal', False)),
            'frame_tensor': frame[0].cpu(),  # Store for visualization
            'masks_up': masks_up.cpu(),       # (K, 1, H, W)
            'entropy_map': ent_up_np,
            'gt_seg': gt_np,
            'materials': materials,
        })
    
    return results


def plot_uncertainty_showcase(results, top_n=5, bottom_n=5):
    """
    高エントロピー vs 低エントロピー シーンの比較図
    """
    sorted_by_entropy = sorted(results, key=lambda x: x['mean_entropy'], reverse=True)
    
    high_ent = sorted_by_entropy[:top_n]
    low_ent = sorted_by_entropy[-bottom_n:]
    
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, top_n, figure=fig, hspace=0.35, wspace=0.15)
    
    # Row 0: High entropy (model uncertain)
    fig.text(0.02, 0.75, 'High\nEntropy\n(Uncertain)', fontsize=16, fontweight='bold',
             color='#D32F2F', ha='center', va='center', rotation=0)
    for i, r in enumerate(high_ent):
        ax = fig.add_subplot(gs[0, i])
        img_np = r['frame_tensor'].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        # Overlay entropy heatmap
        ent = ax.imshow(r['entropy_map'], alpha=0.5, cmap='hot', vmin=0, vmax=1)
        mat_str = f"M:{r['n_metal']} R:{r['n_rubber']}"
        ax.set_title(f"S{r['scene_id']:03d} H̄={r['mean_entropy']:.3f}\n{mat_str} obj={r['n_objects']}",
                     fontsize=12)
        ax.axis('off')
    
    # Row 1: Low entropy (model confident)
    fig.text(0.02, 0.28, 'Low\nEntropy\n(Confident)', fontsize=16, fontweight='bold',
             color='#1565C0', ha='center', va='center', rotation=0)
    for i, r in enumerate(low_ent):
        ax = fig.add_subplot(gs[1, i])
        img_np = r['frame_tensor'].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ent = ax.imshow(r['entropy_map'], alpha=0.5, cmap='hot', vmin=0, vmax=1)
        mat_str = f"M:{r['n_metal']} R:{r['n_rubber']}"
        ax.set_title(f"S{r['scene_id']:03d} H̄={r['mean_entropy']:.3f}\n{mat_str} obj={r['n_objects']}",
                     fontsize=12)
        ax.axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(ent, cax=cbar_ax, label='Normalized Entropy')
    
    fig.suptitle('Mask Entropy Analysis: Where Does the Model Struggle?',
                 fontsize=20, fontweight='bold', y=0.98)
    
    out = FIG_DIR / 'uncertainty_showcase.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {out}")


def plot_entropy_vs_attributes(results):
    """
    エントロピーと物体属性（金属数, 物体数）の散布図
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    ents = [r['mean_entropy'] for r in results]
    n_objs = [r['n_objects'] for r in results]
    n_metals = [r['n_metal'] for r in results]
    high_ratios = [r['high_entropy_ratio'] for r in results]
    has_metal = [r['has_metal'] for r in results]
    
    # 1. Entropy vs #objects
    ax = axes[0]
    colors = ['#E91E63' if m else '#2196F3' for m in has_metal]
    ax.scatter(n_objs, ents, c=colors, alpha=0.5, s=30, edgecolors='none')
    ax.set_xlabel('Number of Objects', fontsize=14)
    ax.set_ylabel('Mean Mask Entropy', fontsize=14)
    ax.set_title('Entropy vs. Object Count', fontsize=15)
    ax.grid(True, alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#E91E63', label='Has Metal'),
                    Patch(facecolor='#2196F3', label='Rubber Only')]
    ax.legend(handles=legend_elems, fontsize=11)
    
    # 2. Entropy vs #metal objects
    ax = axes[1]
    ax.scatter(n_metals, ents, c='#FF9800', alpha=0.5, s=30, edgecolors='none')
    # Add box plot overlay
    unique_metals = sorted(set(n_metals))
    bdata = [[ents[i] for i in range(len(ents)) if n_metals[i] == m] for m in unique_metals]
    bp = ax.boxplot(bdata, positions=unique_metals, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='#FFF3E0', alpha=0.7))
    ax.set_xlabel('Number of Metal Objects', fontsize=14)
    ax.set_ylabel('Mean Mask Entropy', fontsize=14)
    ax.set_title('Entropy vs. Metal Count', fontsize=15)
    ax.grid(True, alpha=0.3)
    
    # 3. High-entropy pixel ratio histogram: metal vs rubber
    ax = axes[2]
    metal_ratios = [r['high_entropy_ratio'] for r in results if r['has_metal']]
    rubber_ratios = [r['high_entropy_ratio'] for r in results if not r['has_metal']]
    ax.hist(rubber_ratios, bins=30, alpha=0.6, color='#2196F3', label=f'Rubber only (n={len(rubber_ratios)})')
    ax.hist(metal_ratios, bins=30, alpha=0.6, color='#E91E63', label=f'Has metal (n={len(metal_ratios)})')
    ax.set_xlabel('High-Entropy Pixel Ratio (>0.7)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Uncertainty Distribution by Material', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Model Uncertainty Correlates with Scene Attributes',
                 fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / 'uncertainty_attributes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {out}")


def plot_detailed_uncertainty(results, n_examples=4):
    """
    高エントロピーシーンの詳細分析：
    入力画像 / GT / マスク全K枚 / エントロピーマップ
    """
    sorted_by_entropy = sorted(results, key=lambda x: x['mean_entropy'], reverse=True)
    examples = sorted_by_entropy[:n_examples]
    
    K = examples[0]['masks_up'].shape[0]
    ncols = 3 + K  # image, GT, entropy, + K masks
    fig, axes = plt.subplots(n_examples, ncols, figsize=(3 * ncols, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]
    
    for row, r in enumerate(examples):
        # Input image
        ax = axes[row, 0]
        img_np = r['frame_tensor'].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"S{r['scene_id']:03d}\nInput", fontsize=11)
        ax.axis('off')
        
        # GT segmentation
        ax = axes[row, 1]
        ax.imshow(r['gt_seg'], cmap='tab10', interpolation='nearest')
        mat_str = ', '.join(r['materials'][:5]) if r['materials'] else '?'
        ax.set_title(f"GT ({r['n_objects']} obj)\n{mat_str}", fontsize=10)
        ax.axis('off')
        
        # Entropy map
        ax = axes[row, 2]
        im = ax.imshow(r['entropy_map'], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f"Entropy\nH̄={r['mean_entropy']:.3f}", fontsize=11)
        ax.axis('off')
        
        # Individual slot masks
        masks = r['masks_up']  # (K, 1, H, W)
        for k in range(K):
            ax = axes[row, 3 + k]
            ax.imshow(masks[k, 0].numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Slot {k}", fontsize=10)
            ax.axis('off')
    
    fig.suptitle('Detailed Mask Decomposition: Highest-Entropy Scenes',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = FIG_DIR / 'uncertainty_detailed.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {out}")


def save_csv(results):
    """エントロピー統計をCSVに保存"""
    import csv
    out = ROOT / 'logs' / 'uncertainty_results.csv'
    keys = ['scene_id', 'mean_entropy', 'max_entropy', 'std_entropy',
            'high_entropy_ratio', 'n_objects', 'n_metal', 'n_rubber', 'has_metal']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})
    print(f"✓ {out}")


if __name__ == '__main__':
    print("=== Uncertainty Analysis ===")
    print(f"Device: {DEVICE}")
    
    # Load best model: 448px τ=0.3 K=11 (FG-ARI 0.696)
    # But for entropy visualization at pixel level, 224px is clearer
    # Use K=9 τ=0.5 224px (FG-ARI 0.658) as the main analysis model
    ckpt_path = ROOT / 'checkpoints' / 'k_sweep' / 'dinov2_vits14_K9' / 'dinov2_vits14' / 'best_model.pt'
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SAViDinosaur(num_slots=9, backbone='dinov2_vits14', image_size=224)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE).eval()
    print(f"  epoch={ckpt.get('epoch', -1)}")
    
    # Run analysis
    results = analyze_all_scenes(model, image_size=224)
    print(f"\nAnalyzed {len(results)} scenes")
    
    ents = [r['mean_entropy'] for r in results]
    print(f"Mean entropy: {np.mean(ents):.4f} ± {np.std(ents):.4f}")
    print(f"Range: [{np.min(ents):.4f}, {np.max(ents):.4f}]")
    
    # Metal vs Rubber
    metal_ents = [r['mean_entropy'] for r in results if r['has_metal']]
    rubber_ents = [r['mean_entropy'] for r in results if not r['has_metal']]
    print(f"\nMetal scenes: {np.mean(metal_ents):.4f} ± {np.std(metal_ents):.4f} (n={len(metal_ents)})")
    print(f"Rubber only:  {np.mean(rubber_ents):.4f} ± {np.std(rubber_ents):.4f} (n={len(rubber_ents)})")
    
    # Generate figures
    print("\n--- Generating figures ---")
    plot_uncertainty_showcase(results, top_n=5, bottom_n=5)
    plot_entropy_vs_attributes(results)
    plot_detailed_uncertainty(results, n_examples=4)
    save_csv(results)
    
    print("\n✅ Uncertainty analysis complete!")
