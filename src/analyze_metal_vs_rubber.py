"""
Metal vs Rubber 時間的一貫性分析
================================

鏡面反射（Metal）vs マット（Rubber）で、
スロットの追跡性能に違いがあるかを分析
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント
from pathlib import Path
import os

from savi_dinosaur import SAViDinosaur


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """学習済みモデルを読み込む"""
    model = SAViDinosaur(num_slots=5, feat_dim=384)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    return model


def compute_temporal_consistency(masks: torch.Tensor) -> dict:
    """
    時間方向のスロット一貫性を計算
    
    Args:
        masks: (T, K, H, W) マスク
    
    Returns:
        dict with metrics:
        - slot_iou: 連続フレーム間のIoU（スロットごと）
        - slot_stability: マスクの変動率
        - assignment_changes: スロット割り当ての変化回数
    """
    t, k, h, w = masks.shape
    
    # 各スロットの連続フレーム間IoU
    slot_ious = []
    for slot_idx in range(k):
        ious = []
        for t_idx in range(t - 1):
            mask_t = (masks[t_idx, slot_idx] > 0.5).float()
            mask_t1 = (masks[t_idx + 1, slot_idx] > 0.5).float()
            
            intersection = (mask_t * mask_t1).sum()
            union = ((mask_t + mask_t1) > 0).float().sum()
            
            iou = (intersection / (union + 1e-8)).item()
            ious.append(iou)
        
        slot_ious.append(np.mean(ious) if ious else 0)
    
    # マスクの変動率（フレーム間の差分）
    stability = []
    for slot_idx in range(k):
        diffs = []
        for t_idx in range(t - 1):
            diff = torch.abs(masks[t_idx, slot_idx] - masks[t_idx + 1, slot_idx]).mean()
            diffs.append(diff.item())
        stability.append(np.mean(diffs) if diffs else 0)
    
    # 各ピクセルで最大スロットが変わった回数
    argmax_masks = masks.argmax(dim=1)  # (T, H, W)
    changes = 0
    for t_idx in range(t - 1):
        changes += (argmax_masks[t_idx] != argmax_masks[t_idx + 1]).float().sum().item()
    assignment_change_rate = changes / (t - 1) / (h * w)
    
    return {
        'slot_ious': slot_ious,
        'mean_iou': np.mean(slot_ious),
        'stability': stability,
        'mean_stability': np.mean(stability),  # 低いほど安定
        'assignment_change_rate': assignment_change_rate,  # 低いほど安定
    }


def analyze_sample(model, data_path: str, device: str = 'cuda') -> dict:
    """単一サンプルを分析"""
    data = torch.load(data_path, weights_only=False)
    video = data['video'][:12]  # 12フレーム使用
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        video_tensor = video.unsqueeze(0).to(device)
        result = model.forward_video(video_tensor)
        masks = result['all_masks'][0].squeeze(2)  # (T, K, H, W)
    
    metrics = compute_temporal_consistency(masks.cpu())
    metrics['filename'] = Path(data_path).name
    metrics['is_metal'] = 'metal' in Path(data_path).name
    
    return metrics


def main():
    print("="*60)
    print("Metal vs Rubber 時間的一貫性分析")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # モデル読み込み
    print("\n1. Loading model...")
    model = load_model('../checkpoints/best_model.pt', device)
    
    # データディレクトリ
    data_dir = Path('../data/movi_a_subset')
    metal_files = sorted(data_dir.glob('metal_*.pt'))
    mixed_files = sorted(data_dir.glob('mixed_*.pt'))
    
    print(f"\n2. Analyzing samples...")
    print(f"   Metal samples: {len(metal_files)}")
    print(f"   Mixed samples: {len(mixed_files)}")
    
    # 分析
    metal_results = []
    mixed_results = []
    
    for f in metal_files[:10]:  # 最初の10サンプル
        result = analyze_sample(model, str(f), device)
        metal_results.append(result)
    
    for f in mixed_files[:10]:
        result = analyze_sample(model, str(f), device)
        mixed_results.append(result)
    
    # 集計
    print("\n" + "="*60)
    print("結果")
    print("="*60)
    
    metal_ious = [r['mean_iou'] for r in metal_results]
    mixed_ious = [r['mean_iou'] for r in mixed_results]
    
    metal_stability = [r['mean_stability'] for r in metal_results]
    mixed_stability = [r['mean_stability'] for r in mixed_results]
    
    metal_changes = [r['assignment_change_rate'] for r in metal_results]
    mixed_changes = [r['assignment_change_rate'] for r in mixed_results]
    
    print(f"\n{'指標':<25} {'Metal':<15} {'Mixed':<15} {'差':<10}")
    print("-" * 65)
    print(f"{'Mean IoU (高いほど良い)':<25} {np.mean(metal_ious):.4f} ± {np.std(metal_ious):.4f}  {np.mean(mixed_ious):.4f} ± {np.std(mixed_ious):.4f}  {np.mean(metal_ious) - np.mean(mixed_ious):+.4f}")
    print(f"{'Stability (低いほど良い)':<25} {np.mean(metal_stability):.4f} ± {np.std(metal_stability):.4f}  {np.mean(mixed_stability):.4f} ± {np.std(mixed_stability):.4f}  {np.mean(metal_stability) - np.mean(mixed_stability):+.4f}")
    print(f"{'Change Rate (低いほど良い)':<25} {np.mean(metal_changes):.4f} ± {np.std(metal_changes):.4f}  {np.mean(mixed_changes):.4f} ± {np.std(mixed_changes):.4f}  {np.mean(metal_changes) - np.mean(mixed_changes):+.4f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # IoU比較
    axes[0].bar(['Metal', 'Mixed'], [np.mean(metal_ious), np.mean(mixed_ious)],
                yerr=[np.std(metal_ious), np.std(mixed_ious)], capsize=5)
    axes[0].set_ylabel('Mean IoU')
    axes[0].set_title('時間的一貫性 (IoU)')
    axes[0].set_ylim(0, 1)
    
    # Stability比較
    axes[1].bar(['Metal', 'Mixed'], [np.mean(metal_stability), np.mean(mixed_stability)],
                yerr=[np.std(metal_stability), np.std(mixed_stability)], capsize=5)
    axes[1].set_ylabel('Mean Stability')
    axes[1].set_title('マスク変動率 (低いほど安定)')
    
    # Change Rate比較
    axes[2].bar(['Metal', 'Mixed'], [np.mean(metal_changes), np.mean(mixed_changes)],
                yerr=[np.std(metal_changes), np.std(mixed_changes)], capsize=5)
    axes[2].set_ylabel('Change Rate')
    axes[2].set_title('スロット割り当て変化率')
    
    plt.tight_layout()
    save_path = '../checkpoints/metal_vs_mixed_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {save_path}")
    
    # 結論
    print("\n" + "="*60)
    print("結論")
    print("="*60)
    
    iou_diff = np.mean(metal_ious) - np.mean(mixed_ious)
    if abs(iou_diff) < 0.05:
        print("→ Metal と Mixed で時間的一貫性に大きな差はない")
    elif iou_diff > 0:
        print(f"→ Metal の方が時間的一貫性が高い (+{iou_diff:.3f} IoU)")
    else:
        print(f"→ Mixed の方が時間的一貫性が高い ({iou_diff:.3f} IoU)")


if __name__ == "__main__":
    main()
