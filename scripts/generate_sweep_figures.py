#!/usr/bin/env python3
"""
K sweep結果の可視化
- Figure: FG-ARI vs K for 3 backbones (line chart)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'docs' / 'paper' / 'figures'


def load_results():
    """sweep_results.csv からK sweep結果を読み込む"""
    csv_path = ROOT / 'logs' / 'sweep_results.csv'
    results = {}  # backbone -> {K: fg_ari}
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['experiment'] in ('k_sweep', 'existing_K11'):
                bb = row['backbone']
                K = int(row['K'])
                fg_ari = float(row['fg_ari'])
                if bb not in results:
                    results[bb] = {}
                # Use k_sweep result if both exist (more consistent training)
                if K not in results[bb] or row['experiment'] == 'k_sweep':
                    results[bb][K] = fg_ari
    
    return results


def load_tau_results():
    """sweep_results.csv からτ sweep結果を読み込む"""
    csv_path = ROOT / 'logs' / 'sweep_results.csv'
    results = {}  # image_size -> {tau: fg_ari}
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['experiment'] == 'tau_sweep':
                img_size = int(row['image_size'])
                tau = float(row['tau'])
                fg_ari = float(row['fg_ari'])
                if img_size not in results:
                    results[img_size] = {}
                results[img_size][tau] = fg_ari
    
    return results


def generate_k_sweep_figure():
    """K sweep: FG-ARI vs K, 3バックボーン比較"""
    results = load_results()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    backbone_config = {
        'dinov2_vits14': {'label': 'DINOv2 ViT-S/14', 'color': '#2196F3', 'marker': 'o'},
        'dino_vits16':   {'label': 'DINOv1 ViT-S/16', 'color': '#FF9800', 'marker': 's'},
        'clip_vitb16':   {'label': 'CLIP ViT-B/16',   'color': '#4CAF50', 'marker': '^'},
    }
    
    for bb, cfg in backbone_config.items():
        if bb not in results:
            continue
        ks = sorted(results[bb].keys())
        aris = [results[bb][k] for k in ks]
        
        ax.plot(ks, aris, marker=cfg['marker'], label=cfg['label'],
                color=cfg['color'], linewidth=2.5, markersize=10, zorder=3)
        
        # Annotate best point
        best_idx = np.argmax(aris)
        ax.annotate(f'{aris[best_idx]:.3f}',
                    xy=(ks[best_idx], aris[best_idx]),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold',
                    color=cfg['color'])
    
    ax.set_xlabel('Number of Slots (K)', fontsize=14)
    ax.set_ylabel('FG-ARI', fontsize=14)
    ax.set_title('FG-ARI vs. Slot Count K across Backbones', fontsize=15)
    ax.set_xticks([3, 5, 7, 9, 11, 13])
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.75)
    
    fig.tight_layout()
    out_path = OUT_DIR / 'k_sweep_fg_ari.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ K sweep figure: {out_path}")


def generate_tau_sweep_figure():
    """τ sweep: FG-ARI vs τ, 224px and 448px"""
    results = load_tau_results()
    
    if not results:
        print("No τ sweep results yet, skipping figure.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    size_config = {
        224: {'label': '224×224 (16×16 tokens)', 'color': '#2196F3', 'marker': 'o'},
        448: {'label': '448×448 (32×32 tokens)', 'color': '#F44336', 'marker': 's'},
    }
    
    for img_size, cfg in size_config.items():
        if img_size not in results:
            continue
        taus = sorted(results[img_size].keys())
        aris = [results[img_size][t] for t in taus]
        
        ax.plot(taus, aris, marker=cfg['marker'], label=cfg['label'],
                color=cfg['color'], linewidth=2.5, markersize=10, zorder=3)
        
        best_idx = np.argmax(aris)
        ax.annotate(f'{aris[best_idx]:.3f}',
                    xy=(taus[best_idx], aris[best_idx]),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold',
                    color=cfg['color'])
    
    ax.set_xlabel('Mask Temperature τ', fontsize=14)
    ax.set_ylabel('FG-ARI', fontsize=14)
    ax.set_title('Effect of Temperature τ on FG-ARI (DINOv2 K=11)', fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.75)
    
    fig.tight_layout()
    out_path = OUT_DIR / 'tau_sweep_fg_ari.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ τ sweep figure: {out_path}")


if __name__ == '__main__':
    generate_k_sweep_figure()
    generate_tau_sweep_figure()
