#!/usr/bin/env python3
"""
SAVi-DINOSAUR アーキテクチャ図を生成
論文挿入用: 左→右のデータフロー図
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color, fontsize=11, text_color='white',
             subtext=None, subsize=9):
    """角丸ボックスを描画"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='black', linewidth=1.2,
                         zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.12, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(x, y - 0.15, subtext, ha='center', va='center',
                fontsize=subsize, color=text_color, zorder=4, style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
    return box


def draw_arrow(ax, x1, y1, x2, y2, label=None, fontsize=9, color='#333333'):
    """矢印を描画"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, connectionstyle='arc3,rad=0'),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.18
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=fontsize, color='#444444', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


def generate_architecture_figure():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-1.8, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ─── Color palette ───
    C_INPUT  = '#4CAF50'   # green
    C_BACKBONE = '#2196F3' # blue
    C_PROJ   = '#FF9800'   # orange
    C_SLOT   = '#9C27B0'   # purple
    C_UPPROJ = '#FF9800'   # orange
    C_DECODER = '#F44336'  # red
    C_LOSS   = '#607D8B'   # gray

    # ─── Row 1 (main pipeline, y=1.0) ───
    y_main = 1.0
    bw, bh = 1.6, 0.8  # box width, height

    # Input image
    draw_box(ax, 0.5, y_main, 1.3, bh, 'Input Image', C_INPUT,
             subtext='(B, 3, 224, 224)', fontsize=10, subsize=8)

    # Frozen ViT Backbone
    draw_box(ax, 2.8, y_main, 1.8, bh, 'Frozen ViT', C_BACKBONE,
             subtext='DINOv2 / v1 / CLIP', fontsize=11, subsize=8)

    # Feature map
    draw_box(ax, 5.2, y_main, 1.7, bh, 'Feature Map', '#78909C',
             subtext='(B, 384, 16, 16)', fontsize=10, subsize=8,
             text_color='white')

    # MLP Projection
    draw_box(ax, 7.4, y_main, 1.5, bh, 'MLP Proj.', C_PROJ,
             subtext='384 → 64', fontsize=10, subsize=9)

    # Slot Attention
    draw_box(ax, 9.7, y_main, 1.8, 1.0, 'Slot\nAttention', C_SLOT,
             fontsize=12, subtext=None)

    # Slots output
    draw_box(ax, 12.0, y_main, 1.3, bh, 'Slots', C_SLOT,
             subtext='(B, K, 64)', fontsize=10, subsize=8,
             text_color='white')

    # Arrows (main path)
    draw_arrow(ax, 1.15, y_main, 1.9, y_main, fontsize=8)
    draw_arrow(ax, 3.7, y_main, 4.35, y_main, fontsize=8)
    draw_arrow(ax, 6.05, y_main, 6.65, y_main,
               label='flatten\n(B, 256, 384)', fontsize=7)
    draw_arrow(ax, 8.15, y_main, 8.8, y_main,
               label='(B, 256, 64)', fontsize=7)
    draw_arrow(ax, 10.6, y_main, 11.35, y_main, fontsize=8)

    # ─── Row 2 (decoder path, y=-0.8) ───
    y_dec = -0.8

    # Up-projection
    draw_box(ax, 9.7, y_dec, 1.5, bh, 'MLP Up', C_UPPROJ,
             subtext='64 → 384', fontsize=10, subsize=9)

    # Spatial Broadcast Decoder
    draw_box(ax, 7.0, y_dec, 2.2, bh, 'Spatial Broadcast\nDecoder', C_DECODER,
             fontsize=10, subsize=8, subtext=None)

    # Outputs
    draw_box(ax, 4.2, y_dec - 0.0, 1.5, 0.6, 'Recon. Feat.', '#78909C',
             fontsize=9, subsize=7, subtext='(B, 384, 16, 16)',
             text_color='white')
    draw_box(ax, 4.2, y_dec + 0.8, 1.3, 0.5, 'Masks', '#E91E63',
             fontsize=9, subsize=7, subtext='(B, K, 1, 16, 16)',
             text_color='white')

    # Loss
    draw_box(ax, 2.2, y_dec + 0.0, 1.3, 0.6, 'MSE Loss', C_LOSS,
             fontsize=10, subsize=8, text_color='white')

    # Arrows (decoder path)
    draw_arrow(ax, 12.0, y_main - 0.4, 12.0, y_dec + 0.8)  # slots down
    ax.annotate('', xy=(10.45, y_dec), xytext=(12.0, y_dec),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8),
                zorder=2)
    draw_arrow(ax, 8.95, y_dec, 8.1, y_dec, fontsize=8)
    # Decoder to recon feat + masks
    draw_arrow(ax, 5.9, y_dec, 4.95, y_dec, label='feat+mask', fontsize=7)
    draw_arrow(ax, 5.9, y_dec + 0.2, 4.85, y_dec + 0.6, fontsize=7)
    # Recon feat → Loss
    draw_arrow(ax, 3.45, y_dec, 2.85, y_dec, fontsize=7)
    # Target feat → Loss (from feature map)
    draw_arrow(ax, 5.2, y_main - 0.4, 5.2, y_dec + 0.55, color='#999999')
    ax.annotate('', xy=(2.85, y_dec + 0.15), xytext=(5.2, y_dec + 0.55),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.2,
                                linestyle='dashed'),
                zorder=2)
    ax.text(3.8, y_dec + 0.55, 'target (detach)', fontsize=7, color='#888888',
            ha='center', style='italic')

    # ─── Slot Attention detail annotation ───
    ax.text(9.7, y_main + 0.72, 'K, V ← features;  Q ← slots',
            ha='center', va='center', fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5',
                      edgecolor='#CE93D8', alpha=0.9))
    ax.text(9.7, y_main - 0.72, 'iters=5, GRU update',
            ha='center', va='center', fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5',
                      edgecolor='#CE93D8', alpha=0.9))

    # ─── Backbone details (annotation) ───
    y_bb = 2.3
    bb_info = [
        ('DINOv2 ViT-S/14', '384-dim, patch=14, 16×16 tokens'),
        ('DINOv1 ViT-S/16', '384-dim, patch=16, 14×14 tokens'),
        ('CLIP ViT-B/16',   '768→384 proj, patch=16, 14×14 tokens'),
    ]
    for i, (name, detail) in enumerate(bb_info):
        xp = 1.0 + i * 4.0
        ax.text(xp, y_bb, f'{name}: {detail}', fontsize=8,
                color='#1565C0', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#E3F2FD',
                          edgecolor='#90CAF9', alpha=0.9))

    # ─── Temperature annotation ───
    ax.text(7.0, y_dec - 0.55, 'τ=0.5 (mask logits / τ → softmax)',
            ha='center', fontsize=8, color='#C62828', style='italic')

    # Title
    ax.text(6.5, 2.65, 'SAVi-DINOSAUR Architecture (Static Image Mode)',
            ha='center', fontsize=14, fontweight='bold', color='#212121')

    fig.tight_layout()
    out_path = OUT_DIR / 'architecture_diagram.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Architecture diagram: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    generate_architecture_figure()
