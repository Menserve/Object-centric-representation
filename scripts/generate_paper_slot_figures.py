#!/usr/bin/env python3
"""
第2報用スロットマスク可視化スクリプト
- Figure 1: DINOv2 K=11 マルチシーン可視化（3シーン × {入力, GT, Top-6スロット}）
- Figure 2: 3バックボーン比較 K=11（1シーン × 3行）
- Figure 3: K=5 vs K=11 比較（DINOv2, 同一シーンで並置）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from savi_dinosaur import SAViDinosaur

# ─── 設定 ─────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'movi_a_v2'
OUT_DIR = ROOT / 'docs' / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10色パレット（スロット用、背景含む）
SLOT_COLORS = np.array([
    [0.12, 0.47, 0.71],  # blue
    [1.00, 0.50, 0.05],  # orange
    [0.17, 0.63, 0.17],  # green
    [0.84, 0.15, 0.16],  # red
    [0.58, 0.40, 0.74],  # purple
    [0.55, 0.34, 0.29],  # brown
    [0.89, 0.47, 0.76],  # pink
    [0.50, 0.50, 0.50],  # gray
    [0.74, 0.74, 0.13],  # olive
    [0.09, 0.75, 0.81],  # cyan
    [0.40, 0.60, 0.20],  # dark green
])


def load_model(ckpt_path, backbone, num_slots, image_size=224):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SAViDinosaur(num_slots=num_slots, backbone=backbone, image_size=image_size)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE).eval()
    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', float('nan'))
    print(f"  ✓ {backbone} K={num_slots}: epoch={epoch}, loss={loss:.4f}")
    return model


def load_scene(scene_idx):
    """シーンを読み込み、224×224にリサイズした画像とGTマスクを返す"""
    path = DATA_DIR / f'scene_{scene_idx:04d}.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    # video: (T, C, H, W) or (T, H, W, C) — 最初のフレームを使用
    video = data['video']
    if video.dim() == 4:
        frame = video[0]
    else:
        frame = video
    
    # CHW化
    if frame.shape[0] not in [1, 3]:
        frame = frame.permute(2, 0, 1)  # HWC → CHW
    
    # float化
    if frame.dtype == torch.uint8:
        frame = frame.float() / 255.0
    
    # 224×224にリサイズ
    frame_224 = F.interpolate(frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
    
    # GT segmentation mask
    gt_mask = None
    if 'segmentations' in data:
        seg = data['segmentations']
        if seg.dim() >= 3:
            gt_seg = seg[0] if seg.dim() == 4 else seg  # first frame
            if gt_seg.dim() == 3:
                gt_seg = gt_seg[0]  # (H, W)
            gt_mask = F.interpolate(
                gt_seg.float().unsqueeze(0).unsqueeze(0), 
                size=(224, 224), mode='nearest'
            )[0, 0].long()
    
    return frame_224, gt_mask


@torch.no_grad()
def get_masks(model, frame_224):
    """モデルからスロットマスクを取得（224×224にアップサンプル済み）"""
    img = frame_224.unsqueeze(0).to(DEVICE)
    recon_feat, target_feat, masks, slots = model.forward_image(img)
    # masks: (1, K, 1, h, w) — feature resolution
    masks = masks[0]  # (K, 1, h, w)
    # 224×224にアップサンプル
    masks_up = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False)
    masks_up = masks_up[:, 0]  # (K, 224, 224)
    return masks_up.cpu()


def make_segmentation_overlay(img_np, masks_np, alpha=0.55):
    """
    マスクargmaxで各ピクセルにスロット色を付与し、入力画像とブレンド
    img_np: (H, W, 3) [0,1]
    masks_np: (K, H, W) softmax済みマスク
    """
    K, H, W = masks_np.shape
    seg = np.argmax(masks_np, axis=0)  # (H, W)
    
    color_map = np.zeros((H, W, 3))
    for k in range(K):
        color_idx = k % len(SLOT_COLORS)
        color_map[seg == k] = SLOT_COLORS[color_idx]
    
    overlay = (1 - alpha) * img_np + alpha * color_map
    return np.clip(overlay, 0, 1)


def make_gt_overlay(img_np, gt_mask_np, alpha=0.55):
    """GT mask をカラーで重畳"""
    H, W = gt_mask_np.shape
    unique_ids = np.unique(gt_mask_np)
    color_map = np.zeros((H, W, 3))
    for i, uid in enumerate(unique_ids):
        if uid == 0:
            continue  # 背景はスキップ
        color_idx = (i - 1) % len(SLOT_COLORS)
        color_map[gt_mask_np == uid] = SLOT_COLORS[color_idx]
    
    overlay = (1 - alpha) * img_np + alpha * color_map
    return np.clip(overlay, 0, 1)


def get_active_slot_indices(masks_np, top_n=6):
    """マスク面積が大きい上位 top_n スロットのインデックスを返す"""
    K = masks_np.shape[0]
    seg = np.argmax(masks_np, axis=0)
    areas = [(seg == k).sum() for k in range(K)]
    sorted_idx = np.argsort(areas)[::-1]
    return sorted_idx[:top_n]


# ═══════════════════════════════════════
# Figure 1: DINOv2 K=11 マルチシーン
# ═══════════════════════════════════════
def figure1_dinov2_multiscene():
    print("\n=== Figure 1: DINOv2 K=11 multi-scene ===")
    model = load_model(
        ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt',
        'dinov2_vits14', num_slots=11
    )
    
    # 3シーン選択（物体数が異なるシーンを選ぶ）
    scene_ids = [5, 42, 100]
    n_scenes = len(scene_ids)
    n_slots_show = 6  # 表示するスロット数
    
    fig, axes = plt.subplots(n_scenes, 2 + n_slots_show, figsize=(18, 6.5),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
    
    for row, sid in enumerate(scene_ids):
        frame, gt_mask = load_scene(sid)
        masks = get_masks(model, frame)
        
        img_np = frame.permute(1, 2, 0).numpy()
        masks_np = masks.numpy()
        
        # Col 0: 入力画像
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=11)
        axes[row, 0].set_ylabel(f'Scene {sid}', fontsize=11)
        axes[row, 0].axis('off')
        
        # Col 1: GT + slot segmentation overlay
        seg_overlay = make_segmentation_overlay(img_np, masks_np)
        axes[row, 1].imshow(seg_overlay)
        axes[row, 1].set_title('Slot Seg.' if row == 0 else '', fontsize=11)
        axes[row, 1].axis('off')
        
        # Cols 2-: 個別スロットマスク（面積上位）
        top_slots = get_active_slot_indices(masks_np, top_n=n_slots_show)
        for j, slot_idx in enumerate(top_slots):
            mask_k = masks_np[slot_idx]
            # 個別マスクをheatmap + 画像ブレンド
            heatmap = plt.cm.hot(mask_k / (mask_k.max() + 1e-8))[:, :, :3]
            blend = 0.4 * img_np + 0.6 * heatmap
            blend = np.clip(blend, 0, 1)
            axes[row, 2 + j].imshow(blend)
            if row == 0:
                axes[row, 2 + j].set_title(f'Slot {slot_idx}', fontsize=11)
            axes[row, 2 + j].axis('off')
    
    fig.suptitle('DINOv2 K=11: Slot Mask Visualization (FG-ARI = 0.470)', fontsize=14, y=0.98)
    fig.savefig(OUT_DIR / 'paper_dinov2_k11_slots.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {OUT_DIR / 'paper_dinov2_k11_slots.png'}")


# ═══════════════════════════════════════
# Figure 2: 3バックボーン比較 K=11
# ═══════════════════════════════════════
def figure2_3backbone_comparison():
    print("\n=== Figure 2: 3-backbone K=11 comparison ===")
    configs = [
        ('DINOv2 (FG-ARI=0.470)', ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt', 'dinov2_vits14'),
        ('DINOv1 (FG-ARI=0.131)', ROOT / 'checkpoints' / 'dinov1_K11' / 'dino_vits16' / 'best_model.pt', 'dino_vits16'),
        ('CLIP (FG-ARI=0.110)',    ROOT / 'checkpoints' / 'clip_K11' / 'clip_vitb16' / 'best_model.pt', 'clip_vitb16'),
    ]
    
    models = {}
    for name, ckpt_path, backbone in configs:
        models[name] = load_model(ckpt_path, backbone, num_slots=11)
    
    scene_id = 42
    frame, gt_mask = load_scene(scene_id)
    img_np = frame.permute(1, 2, 0).numpy()
    
    n_slots_show = 6
    fig, axes = plt.subplots(3, 2 + n_slots_show, figsize=(18, 7),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.20})
    
    for row, (name, model) in enumerate(models.items()):
        masks = get_masks(model, frame)
        masks_np = masks.numpy()
        
        # Col 0: 入力 (only show once semantically, but fill all rows)
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=11)
        axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Col 1: slot segmentation
        seg_overlay = make_segmentation_overlay(img_np, masks_np)
        axes[row, 1].imshow(seg_overlay)
        axes[row, 1].set_title('Slot Seg.' if row == 0 else '', fontsize=11)
        axes[row, 1].axis('off')
        
        # Cols 2-: 個別スロット
        top_slots = get_active_slot_indices(masks_np, top_n=n_slots_show)
        for j, slot_idx in enumerate(top_slots):
            mask_k = masks_np[slot_idx]
            heatmap = plt.cm.hot(mask_k / (mask_k.max() + 1e-8))[:, :, :3]
            blend = 0.4 * img_np + 0.6 * heatmap
            blend = np.clip(blend, 0, 1)
            axes[row, 2 + j].imshow(blend)
            if row == 0:
                axes[row, 2 + j].set_title(f'Slot {slot_idx}', fontsize=11)
            axes[row, 2 + j].axis('off')
    
    fig.suptitle(f'3-Backbone Comparison at K=11 (Scene {scene_id})', fontsize=14, y=0.98)
    fig.savefig(OUT_DIR / 'paper_3backbone_k11_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {OUT_DIR / 'paper_3backbone_k11_comparison.png'}")


# ═══════════════════════════════════════
# Figure 3: K=5 vs K=11 (DINOv2)
# ═══════════════════════════════════════
def figure3_k5_vs_k11():
    print("\n=== Figure 3: DINOv2 K=5 vs K=11 ===")
    
    # K=5 モデル（300サンプル版、現行アーキテクチャ互換）
    model_k5 = load_model(
        ROOT / 'checkpoints' / 'dinov2_v2_300samples' / 'dinov2_vits14' / 'best_model.pt',
        'dinov2_vits14', num_slots=5
    )
    # K=11 モデル
    model_k11 = load_model(
        ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt',
        'dinov2_vits14', num_slots=11
    )
    
    scene_ids = [5, 42]
    fig, axes = plt.subplots(len(scene_ids) * 2, 8, figsize=(18, 9),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    
    for s_idx, sid in enumerate(scene_ids):
        frame, gt_mask = load_scene(sid)
        img_np = frame.permute(1, 2, 0).numpy()
        
        for k_idx, (model, K, label) in enumerate([
            (model_k5, 5, f'K=5 (FG-ARI=0.168)'),
            (model_k11, 11, f'K=11 (FG-ARI=0.470)')
        ]):
            row = s_idx * 2 + k_idx
            masks = get_masks(model, frame)
            masks_np = masks.numpy()
            
            # Col 0: Input
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')
            if row == 0:
                axes[row, 0].set_title('Input', fontsize=10)
            axes[row, 0].axis('off')
            
            # Col 1: Slot segmentation
            seg_overlay = make_segmentation_overlay(img_np, masks_np)
            axes[row, 1].imshow(seg_overlay)
            if row == 0:
                axes[row, 1].set_title('Slot Seg.', fontsize=10)
            axes[row, 1].axis('off')
            
            # Cols 2-7: Top-6 slots (or all 5 for K=5)
            n_show = min(6, K)
            top_slots = get_active_slot_indices(masks_np, top_n=n_show)
            for j in range(6):
                if j < n_show:
                    slot_idx = top_slots[j]
                    mask_k = masks_np[slot_idx]
                    heatmap = plt.cm.hot(mask_k / (mask_k.max() + 1e-8))[:, :, :3]
                    blend = 0.4 * img_np + 0.6 * heatmap
                    blend = np.clip(blend, 0, 1)
                    axes[row, 2 + j].imshow(blend)
                    if row == 0:
                        axes[row, 2 + j].set_title(f'Slot {slot_idx}', fontsize=10)
                else:
                    axes[row, 2 + j].axis('off')
                axes[row, 2 + j].axis('off')
    
    # シーン区切り線をテキストで
    fig.text(0.01, 0.75, f'Scene {scene_ids[0]}', fontsize=12, rotation=90, va='center', fontweight='bold')
    fig.text(0.01, 0.30, f'Scene {scene_ids[1]}', fontsize=12, rotation=90, va='center', fontweight='bold')
    
    fig.suptitle('DINOv2: K=5 vs K=11 Slot Differentiation', fontsize=14, y=0.98)
    fig.savefig(OUT_DIR / 'paper_k5_vs_k11_slots.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {OUT_DIR / 'paper_k5_vs_k11_slots.png'}")


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")
    
    figure1_dinov2_multiscene()
    figure2_3backbone_comparison()
    figure3_k5_vs_k11()
    
    print("\n✓ All figures generated.")
