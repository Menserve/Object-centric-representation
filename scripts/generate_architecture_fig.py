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


def draw_box(ax, x, y, w, h, text, color, fontsize=18, text_color='white',
             subtext=None, subsize=14):
    """角丸ボックスを描画"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.10",
                         facecolor=color, edgecolor='black', linewidth=1.8,
                         zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.18, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(x, y - 0.22, subtext, ha='center', va='center',
                fontsize=subsize, color=text_color, zorder=4, style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
    return box


def draw_arrow(ax, x1, y1, x2, y2, label=None, fontsize=14, color='#333333'):
    """矢印を描画"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.2, connectionstyle='arc3,rad=0'),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.25
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=fontsize, color='#444444', zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85))


def generate_architecture_figure():
    fig, ax = plt.subplots(1, 1, figsize=(28, 12))
    ax.set_xlim(-1.5, 24.0)
    ax.set_ylim(-4.5, 5.5)
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

    # ─── Row 1 (main pipeline, y=1.8) ───
    y_main = 1.8
    bh = 1.4  # box height

    # Input image
    draw_box(ax, 0.5, y_main, 2.6, bh, 'Input Image', C_INPUT,
             subtext='(B, 3, 224, 224)', fontsize=22, subsize=16)

    # Frozen ViT Backbone
    draw_box(ax, 4.8, y_main, 3.0, bh, 'Frozen ViT', C_BACKBONE,
             subtext='DINOv2 / v1 / CLIP', fontsize=22, subsize=16)

    # Feature map
    draw_box(ax, 9.5, y_main, 3.0, bh, 'Feature Map', '#78909C',
             subtext='(B, 384, 16, 16)', fontsize=22, subsize=16,
             text_color='white')

    # MLP Projection
    draw_box(ax, 14.0, y_main, 2.8, bh, 'MLP Proj.', C_PROJ,
             subtext='384 → 64', fontsize=22, subsize=17)

    # Slot Attention
    draw_box(ax, 18.2, y_main, 3.0, 1.8, 'Slot\nAttention', C_SLOT,
             fontsize=24, subtext=None)

    # Slots output
    draw_box(ax, 22.2, y_main, 2.4, bh, 'Slots', C_SLOT,
             subtext='(B, K, 64)', fontsize=22, subsize=16,
             text_color='white')

    # Arrows (main path)
    draw_arrow(ax, 1.8, y_main, 3.3, y_main, fontsize=16)
    draw_arrow(ax, 6.3, y_main, 8.0, y_main, fontsize=16)
    draw_arrow(ax, 11.0, y_main, 12.6, y_main,
               label='flatten\n(B, 256, 384)', fontsize=15)
    draw_arrow(ax, 15.4, y_main, 16.7, y_main,
               label='(B, 256, 64)', fontsize=15)
    draw_arrow(ax, 19.7, y_main, 21.0, y_main, fontsize=16)

    # ─── Row 2 (decoder path, y=-1.8) ───
    y_dec = -1.8

    # Up-projection
    draw_box(ax, 18.2, y_dec, 2.8, bh, 'MLP Up', C_UPPROJ,
             subtext='64 → 384', fontsize=22, subsize=17)

    # Spatial Broadcast Decoder
    draw_box(ax, 13.0, y_dec, 3.4, bh, 'Spatial Broadcast\nDecoder', C_DECODER,
             fontsize=20, subsize=16, subtext=None)

    # Outputs
    draw_box(ax, 8.0, y_dec, 2.8, 1.2, 'Recon. Feat.', '#78909C',
             fontsize=20, subsize=15, subtext='(B, 384, 16, 16)',
             text_color='white')
    draw_box(ax, 8.0, y_dec + 1.8, 2.4, 1.0, 'Masks', '#E91E63',
             fontsize=20, subsize=14, subtext='(B, K, 1, 16, 16)',
             text_color='white')

    # Loss
    draw_box(ax, 3.8, y_dec, 2.4, 1.2, 'MSE Loss', C_LOSS,
             fontsize=20, subsize=16, text_color='white')

    # Arrows (decoder path)
    draw_arrow(ax, 22.2, y_main - 0.7, 22.2, y_dec + 1.2)  # slots down
    ax.annotate('', xy=(19.6, y_dec), xytext=(22.2, y_dec),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5),
                zorder=2)
    draw_arrow(ax, 16.8, y_dec, 14.7, y_dec, fontsize=16)
    # Decoder to recon feat + masks
    draw_arrow(ax, 11.3, y_dec, 9.4, y_dec, label='feat+mask', fontsize=15)
    draw_arrow(ax, 11.3, y_dec + 0.4, 9.2, y_dec + 1.3, fontsize=15)
    # Recon feat → Loss
    draw_arrow(ax, 6.6, y_dec, 5.0, y_dec, fontsize=15)
    # Target feat → Loss (from feature map)
    draw_arrow(ax, 9.5, y_main - 0.7, 9.5, y_dec + 1.5, color='#999999')
    ax.annotate('', xy=(5.0, y_dec + 0.3), xytext=(9.5, y_dec + 1.5),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.8,
                                linestyle='dashed'),
                zorder=2)
    ax.text(7.0, y_dec + 1.2, 'target (detach)', fontsize=16, color='#888888',
            ha='center', style='italic')

    # ─── Slot Attention detail annotation ───
    ax.text(18.2, y_main + 1.4, 'K, V ← features;  Q ← slots',
            ha='center', va='center', fontsize=17, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5',
                      edgecolor='#CE93D8', alpha=0.9))
    ax.text(18.2, y_main - 1.45, 'iters=5, GRU update',
            ha='center', va='center', fontsize=17, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5',
                      edgecolor='#CE93D8', alpha=0.9))

    # ─── Backbone details (annotation) ───
    y_bb = 4.2
    bb_info = [
        ('DINOv2 ViT-S/14', '384-dim, patch=14, 16×16 tokens'),
        ('DINOv1 ViT-S/16', '384-dim, patch=16, 14×14 tokens'),
        ('CLIP ViT-B/16',   '768→384 proj, patch=16, 14×14 tokens'),
    ]
    for i, (name, detail) in enumerate(bb_info):
        xp = 0.0 + i * 7.5
        ax.text(xp, y_bb, f'{name}: {detail}', fontsize=16,
                color='#1565C0', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#E3F2FD',
                          edgecolor='#90CAF9', alpha=0.9))

    # ─── Temperature annotation ───
    ax.text(13.0, y_dec - 1.1, 'τ=0.5 (mask logits / τ → softmax)',
            ha='center', fontsize=17, color='#C62828', style='italic')

    # Title
    ax.text(11.25, 5.0, 'SAVi-DINOSAUR Architecture (Static Image Mode)',
            ha='center', fontsize=28, fontweight='bold', color='#212121')

    fig.tight_layout()
    out_path = OUT_DIR / 'architecture_diagram.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Architecture diagram: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    generate_architecture_figure()
