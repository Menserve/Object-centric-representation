#!/usr/bin/env python3
"""
全スイープ実験のチェックポイントを評価し、結果をCSVに保存
K sweep + τ sweep 両方に対応
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import csv
import json
from pathlib import Path
from tqdm import tqdm

from savi_dinosaur import SAViDinosaur
from compute_ari_v2 import compute_per_object_material_ari

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'movi_a_v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path, backbone, num_slots, image_size=224):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SAViDinosaur(num_slots=num_slots, backbone=backbone, image_size=image_size)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(DEVICE).eval(), ckpt.get('epoch', -1), ckpt.get('loss', float('nan'))


@torch.no_grad()
def evaluate_all_scenes(model, image_size=224, n_scenes=300):
    """全300シーンでFG-ARI, Metal-ARI, Rubber-ARIを計算"""
    results = []
    
    for sid in range(n_scenes):
        path = DATA_DIR / f'scene_{sid:04d}.pt'
        if not path.exists():
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Frame 0
        video = data['video']
        frame = video[0] if video.dim() == 4 else video
        if frame.shape[0] not in [1, 3]:
            frame = frame.permute(2, 0, 1)
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        frame = F.interpolate(frame.unsqueeze(0), size=(image_size, image_size),
                              mode='bilinear', align_corners=False)
        
        # GT segmentation
        seg = data['segmentation']
        seg_frame = seg[0] if seg.dim() == 3 else seg
        seg_resized = F.interpolate(seg_frame.unsqueeze(0).unsqueeze(0).float(),
                                     size=(image_size, image_size),
                                     mode='nearest').squeeze().long()
        
        # Forward
        img = frame.to(DEVICE)
        _, _, masks, _ = model.forward_image(img)
        masks_np = masks[0, :, 0].cpu().numpy()  # (K, H, W) at feature resolution
        # Upsample masks to image resolution
        masks_up = F.interpolate(masks[0], size=(image_size, image_size),
                                 mode='bilinear', align_corners=False)
        masks_np = masks_up[:, 0].cpu().numpy()  # (K, image_size, image_size)
        
        gt_np = seg_resized.numpy()
        materials = data.get('materials', [])
        
        ari_result = compute_per_object_material_ari(masks_np, gt_np, materials)
        results.append(ari_result)
    
    # Aggregate
    fg_aris = [r['fg_ari'] for r in results if not np.isnan(r['fg_ari'])]
    metal_aris = [r['metal_ari'] for r in results if not np.isnan(r['metal_ari'])]
    rubber_aris = [r['rubber_ari'] for r in results if not np.isnan(r['rubber_ari'])]
    
    return {
        'fg_ari': np.mean(fg_aris) if fg_aris else float('nan'),
        'metal_ari': np.mean(metal_aris) if metal_aris else float('nan'),
        'rubber_ari': np.mean(rubber_aris) if rubber_aris else float('nan'),
        'n_valid': len(fg_aris),
    }


def scan_and_evaluate_k_sweep():
    """K sweep のチェックポイントを全て評価"""
    sweep_dir = ROOT / 'checkpoints' / 'k_sweep'
    if not sweep_dir.exists():
        print("K sweep checkpoints not found, skipping.")
        return []
    
    results = []
    backbones = ['dinov2_vits14', 'dino_vits16', 'clip_vitb16']
    k_values = [3, 5, 7, 9, 11, 13]
    
    for bb in backbones:
        for k in k_values:
            ckpt_path = sweep_dir / f'{bb}_K{k}' / bb / 'best_model.pt'
            if not ckpt_path.exists():
                print(f"  [SKIP] {bb} K={k}")
                continue
            
            print(f"  Evaluating {bb} K={k}...")
            model, epoch, train_loss = load_model(ckpt_path, bb, k)
            ari = evaluate_all_scenes(model, image_size=224)
            
            results.append({
                'experiment': 'k_sweep',
                'backbone': bb,
                'K': k,
                'image_size': 224,
                'tau': 0.5,
                'epoch': epoch,
                'train_loss': train_loss,
                **ari
            })
            print(f"    FG-ARI={ari['fg_ari']:.4f}, Metal={ari['metal_ari']:.4f}, Rubber={ari['rubber_ari']:.4f}")
            
            del model
            torch.cuda.empty_cache()
    
    return results


def scan_and_evaluate_tau_sweep():
    """τ sweep のチェックポイントを全て評価"""
    sweep_dir = ROOT / 'checkpoints' / 'tau_sweep'
    if not sweep_dir.exists():
        print("τ sweep checkpoints not found, skipping.")
        return []
    
    results = []
    bb = 'dinov2_vits14'
    image_sizes = [224, 448]
    tau_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    for img_size in image_sizes:
        for tau in tau_values:
            tau_str = str(tau)
            ckpt_path = sweep_dir / f'{bb}_{img_size}_tau{tau_str}' / bb / 'best_model.pt'
            if not ckpt_path.exists():
                print(f"  [SKIP] {img_size}px τ={tau}")
                continue
            
            print(f"  Evaluating {img_size}px τ={tau}...")
            model, epoch, train_loss = load_model(ckpt_path, bb, 11, image_size=img_size)
            ari = evaluate_all_scenes(model, image_size=img_size)
            
            results.append({
                'experiment': 'tau_sweep',
                'backbone': bb,
                'K': 11,
                'image_size': img_size,
                'tau': tau,
                'epoch': epoch,
                'train_loss': train_loss,
                **ari
            })
            print(f"    FG-ARI={ari['fg_ari']:.4f}, Metal={ari['metal_ari']:.4f}, Rubber={ari['rubber_ari']:.4f}")
            
            del model
            torch.cuda.empty_cache()
    
    return results


def save_results(results, filename):
    """結果をCSVに保存"""
    if not results:
        return
    
    out_path = ROOT / 'logs' / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    keys = results[0].keys()
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved: {out_path}")


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_DIR}")
    
    # Also evaluate existing K=11 checkpoints for reference
    existing_results = []
    for bb, bb_dir in [
        ('dinov2_vits14', 'dinov2_K11/dinov2_vits14'),
        ('dino_vits16', 'dinov1_K11/dino_vits16'),
        ('clip_vitb16', 'clip_K11/clip_vitb16'),
    ]:
        ckpt_path = ROOT / 'checkpoints' / bb_dir / 'best_model.pt'
        if ckpt_path.exists():
            print(f"  Evaluating existing {bb} K=11...")
            model, epoch, train_loss = load_model(ckpt_path, bb, 11)
            ari = evaluate_all_scenes(model, image_size=224)
            existing_results.append({
                'experiment': 'existing_K11',
                'backbone': bb,
                'K': 11,
                'image_size': 224,
                'tau': 0.5,
                'epoch': epoch,
                'train_loss': train_loss,
                **ari
            })
            print(f"    FG-ARI={ari['fg_ari']:.4f}")
            del model
            torch.cuda.empty_cache()
    
    print("\n=== K Sweep Evaluation ===")
    k_results = scan_and_evaluate_k_sweep()
    
    print("\n=== τ Sweep Evaluation ===")
    tau_results = scan_and_evaluate_tau_sweep()
    
    all_results = existing_results + k_results + tau_results
    save_results(all_results, 'sweep_results.csv')
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Experiment':<15} {'Backbone':<16} {'K':>3} {'ImgSz':>5} {'τ':>4} {'FG-ARI':>8} {'Metal':>8} {'Rubber':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<15} {r['backbone']:<16} {r['K']:>3} {r['image_size']:>5} {r['tau']:>4} "
              f"{r['fg_ari']:>8.4f} {r['metal_ari']:>8.4f} {r['rubber_ari']:>8.4f}")
    print("=" * 80)
