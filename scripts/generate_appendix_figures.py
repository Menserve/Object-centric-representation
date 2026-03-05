#!/usr/bin/env python3
"""
付録用スロット可視化スクリプト
- Figure A: DINOv2 K=11 ランダム20シーン（cherry-picking排除）
- Figure B: Soft Mask + エントロピーマップ（不確実性提示）
- Figure C: 3バックボーン × ランダム5シーン比較
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from savi_dinosaur import SAViDinosaur

# ─── 設定 ─────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'movi_a_v2'
OUT_DIR = ROOT / 'docs' / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLOT_COLORS = np.array([
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17],
    [0.84, 0.15, 0.16], [0.58, 0.40, 0.74], [0.55, 0.34, 0.29],
    [0.89, 0.47, 0.76], [0.50, 0.50, 0.50], [0.74, 0.74, 0.13],
    [0.09, 0.75, 0.81], [0.40, 0.60, 0.20],
])

# 再現性のための固定シード
np.random.seed(42)
RANDOM_20 = sorted(np.random.choice(300, size=20, replace=False))
RANDOM_5 = sorted(np.random.choice(300, size=5, replace=False))
print(f"Random 20 scenes: {list(RANDOM_20)}")
print(f"Random 5 scenes: {list(RANDOM_5)}")


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
    path = DATA_DIR / f'scene_{scene_idx:04d}.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    video = data['video']
    frame = video[0] if video.dim() == 4 else video
    if frame.shape[0] not in [1, 3]:
        frame = frame.permute(2, 0, 1)
    if frame.dtype == torch.uint8:
        frame = frame.float() / 255.0
    frame_224 = F.interpolate(frame.unsqueeze(0), size=(224, 224),
                              mode='bilinear', align_corners=False)[0]
    return frame_224


@torch.no_grad()
def get_masks(model, frame_224):
    img = frame_224.unsqueeze(0).to(DEVICE)
    _, _, masks, _ = model.forward_image(img)
    masks = masks[0]  # (K, 1, h, w)
    masks_up = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False)
    masks_up = masks_up[:, 0]  # (K, 224, 224)
    return masks_up.cpu()


def make_seg_overlay(img_np, masks_np, alpha=0.55):
    K, H, W = masks_np.shape
    seg = np.argmax(masks_np, axis=0)
    color_map = np.zeros((H, W, 3))
    for k in range(K):
        color_map[seg == k] = SLOT_COLORS[k % len(SLOT_COLORS)]
    return np.clip((1 - alpha) * img_np + alpha * color_map, 0, 1)


def compute_entropy(masks_np):
    """per-pixel entropy of soft mask distribution"""
    # masks_np: (K, H, W), already softmax
    eps = 1e-8
    ent = -np.sum(masks_np * np.log(masks_np + eps), axis=0)  # (H, W)
    return ent


def get_top_slots(masks_np, top_n=4):
    seg = np.argmax(masks_np, axis=0)
    areas = [(seg == k).sum() for k in range(masks_np.shape[0])]
    return np.argsort(areas)[::-1][:top_n]


# ═══════════════════════════════════════
# Figure A: DINOv2 K=11 ランダム20シーン
# ═══════════════════════════════════════
def figure_a_random20():
    print("\n=== Figure A: DINOv2 K=11, random 20 scenes ===")
    model = load_model(
        ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt',
        'dinov2_vits14', num_slots=11
    )

    n_scenes = len(RANDOM_20)
    # Layout: 20 rows × 6 cols (input, seg, top-4 slots)
    n_cols = 6
    fig, axes = plt.subplots(n_scenes, n_cols, figsize=(12, n_scenes * 1.5),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.08})

    for row, sid in enumerate(RANDOM_20):
        frame = load_scene(sid)
        masks = get_masks(model, frame)
        img_np = frame.permute(1, 2, 0).numpy()
        masks_np = masks.numpy()

        # Col 0: input
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f'{sid}', fontsize=10, rotation=0, labelpad=20)
        if row == 0:
            axes[row, 0].set_title('Input', fontsize=12)
        axes[row, 0].axis('off')

        # Col 1: slot segmentation
        seg_ov = make_seg_overlay(img_np, masks_np)
        axes[row, 1].imshow(seg_ov)
        if row == 0:
            axes[row, 1].set_title('Seg.', fontsize=12)
        axes[row, 1].axis('off')

        # Cols 2-5: top-4 slots (heatmap)
        top_idx = get_top_slots(masks_np, top_n=4)
        for j, slot_idx in enumerate(top_idx):
            mask_k = masks_np[slot_idx]
            hm = plt.cm.hot(mask_k / (mask_k.max() + 1e-8))[:, :, :3]
            blend = np.clip(0.4 * img_np + 0.6 * hm, 0, 1)
            axes[row, 2 + j].imshow(blend)
            if row == 0:
                axes[row, 2 + j].set_title(f'Slot', fontsize=11)
            axes[row, 2 + j].axis('off')

    fig.suptitle('Appendix A: DINOv2 K=11 — Random 20 Scenes (seed=42)',
                 fontsize=14, y=1.0)
    out_path = OUT_DIR / 'appendix_a_random20.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════
# Figure B: Soft Mask + エントロピーマップ
# ═══════════════════════════════════════
def figure_b_softmask_entropy():
    print("\n=== Figure B: Soft Mask + Entropy (4 scenes) ===")
    model = load_model(
        ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt',
        'dinov2_vits14', num_slots=11
    )

    scenes = RANDOM_5[:4]  # 4シーン
    n_top = 4
    # Layout: 4 scenes × (input + hard_seg + entropy + 4 soft masks) = 7 cols
    n_cols = 3 + n_top
    fig, axes = plt.subplots(len(scenes), n_cols, figsize=(14, len(scenes) * 2.5),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.12})

    for row, sid in enumerate(scenes):
        frame = load_scene(sid)
        masks = get_masks(model, frame)
        img_np = frame.permute(1, 2, 0).numpy()
        masks_np = masks.numpy()

        # Col 0: Input
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f'Scene {sid}', fontsize=11)
        if row == 0:
            axes[row, 0].set_title('Input', fontsize=12)
        axes[row, 0].axis('off')

        # Col 1: Hard segmentation (argmax)
        seg_ov = make_seg_overlay(img_np, masks_np)
        axes[row, 1].imshow(seg_ov)
        if row == 0:
            axes[row, 1].set_title('Hard Seg.', fontsize=12)
        axes[row, 1].axis('off')

        # Col 2: Entropy map
        ent = compute_entropy(masks_np)
        max_ent = np.log(masks_np.shape[0])  # max entropy = ln(K)
        im = axes[row, 2].imshow(ent, cmap='inferno', vmin=0, vmax=max_ent)
        if row == 0:
            axes[row, 2].set_title('Entropy', fontsize=12)
        axes[row, 2].axis('off')

        # Cols 3-6: Individual soft masks (continuous heatmap, no argmax)
        top_idx = get_top_slots(masks_np, top_n=n_top)
        for j, slot_idx in enumerate(top_idx):
            mask_k = masks_np[slot_idx]
            # Pure soft mask: show attention weight directly
            axes[row, 3 + j].imshow(mask_k, cmap='viridis', vmin=0, vmax=1)
            if row == 0:
                axes[row, 3 + j].set_title(f'Soft Slot {slot_idx}', fontsize=11)
            axes[row, 3 + j].axis('off')

    # Colorbar for entropy
    cbar_ax = fig.add_axes([0.33, 0.02, 0.10, 0.012])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Entropy')

    fig.suptitle('Appendix B: Soft Masks & Per-pixel Entropy (DINOv2 K=11)',
                 fontsize=14, y=1.0)
    out_path = OUT_DIR / 'appendix_b_softmask_entropy.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════
# Figure C: 3バックボーン × ランダム5シーン
# ═══════════════════════════════════════
def figure_c_3backbone_random5():
    print("\n=== Figure C: 3-backbone × random 5 scenes ===")
    configs = [
        ('DINOv2', ROOT / 'checkpoints' / 'dinov2_K11' / 'dinov2_vits14' / 'best_model.pt', 'dinov2_vits14'),
        ('DINOv1', ROOT / 'checkpoints' / 'dinov1_K11' / 'dino_vits16' / 'best_model.pt', 'dino_vits16'),
        ('CLIP',   ROOT / 'checkpoints' / 'clip_K11' / 'clip_vitb16' / 'best_model.pt', 'clip_vitb16'),
    ]
    models = {}
    for name, ckpt_path, backbone in configs:
        models[name] = load_model(ckpt_path, backbone, num_slots=11)

    scenes = RANDOM_5
    n_backbones = 3
    n_scenes = len(scenes)
    # Layout: (3 backbones × 5 scenes) rows × 3 cols (input, seg, entropy)
    total_rows = n_backbones * n_scenes
    n_cols = 3
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(7, total_rows * 1.4),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.10})

    backbone_names = list(models.keys())
    for b_idx, (bname, model) in enumerate(models.items()):
        for s_idx, sid in enumerate(scenes):
            row = b_idx * n_scenes + s_idx
            frame = load_scene(sid)
            masks = get_masks(model, frame)
            img_np = frame.permute(1, 2, 0).numpy()
            masks_np = masks.numpy()

            # Col 0: Input
            axes[row, 0].imshow(img_np)
            if s_idx == 0:
                axes[row, 0].set_ylabel(bname, fontsize=11, fontweight='bold')
            else:
                axes[row, 0].set_ylabel(f'  #{sid}', fontsize=9)
            if row == 0:
                axes[row, 0].set_title('Input', fontsize=12)
            axes[row, 0].axis('off')

            # Col 1: Hard segmentation
            seg_ov = make_seg_overlay(img_np, masks_np)
            axes[row, 1].imshow(seg_ov)
            if row == 0:
                axes[row, 1].set_title('Slot Seg.', fontsize=12)
            axes[row, 1].axis('off')

            # Col 2: Entropy
            ent = compute_entropy(masks_np)
            max_ent = np.log(11)
            axes[row, 2].imshow(ent, cmap='inferno', vmin=0, vmax=max_ent)
            if row == 0:
                axes[row, 2].set_title('Entropy', fontsize=12)
            axes[row, 2].axis('off')

        # Draw separator line after each backbone group
        if b_idx < n_backbones - 1:
            sep_y = (b_idx + 1) * n_scenes - 1
            for c in range(n_cols):
                rect = axes[sep_y, c].get_position()
                fig.add_artist(plt.Line2D(
                    [rect.x0, rect.x1], [rect.y0 - 0.005, rect.y0 - 0.005],
                    transform=fig.transFigure, color='gray', linewidth=1.5))

    fig.suptitle('Appendix C: 3-Backbone × 5 Random Scenes (K=11, seed=42)',
                 fontsize=14, y=1.0)
    out_path = OUT_DIR / 'appendix_c_3backbone_random5.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    figure_a_random20()
    figure_b_softmask_entropy()
    figure_c_3backbone_random5()
    print("\n✓ All appendix figures generated.")
