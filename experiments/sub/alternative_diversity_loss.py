"""
さらに強力な多様性正則化の実験
=====================================

もし diversity_weight=1.0 でも改善が不十分な場合、
以下のより攻撃的な設定を試す：

1. diversity_weight を 5.0 or 10.0 に
2. より多くのSlot（5 → 7）
3. Hungarian matching loss（最適割り当て）
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_diversity_loss(masks, gt_masks):
    """
    Hungarian algorithm による最適Slot割り当て
    
    Args:
        masks: (B, K, H, W) 予測マスク
        gt_masks: (B, M, H, W) Ground truthマスク
    
    Returns:
        loss: スカラー
    """
    B, K, H, W = masks.shape
    _, M, _, _ = gt_masks.shape
    
    # コスト行列を計算（IoU距離）
    masks_flat = masks.view(B, K, -1)
    gt_flat = gt_masks.view(B, M, -1)
    
    total_loss = 0
    for b in range(B):
        # (K, M) コスト行列
        cost_matrix = torch.zeros(K, M)
        for k in range(K):
            for m in range(M):
                intersection = (masks_flat[b, k] * gt_flat[b, m]).sum()
                union = ((masks_flat[b, k] + gt_flat[b, m]) > 0).float().sum()
                iou = intersection / (union + 1e-8)
                cost_matrix[k, m] = 1.0 - iou  # IoU距離
        
        # Hungarian algorithm で最適割り当て
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # 割り当てられたSlotとGTの損失
        for k, m in zip(row_ind, col_ind):
            total_loss += cost_matrix[k, m]
    
    return total_loss / B


def contrastive_diversity_loss(slot_features):
    """
    Contrastive learning ベースの多様性損失
    
    Args:
        slot_features: (B, K, D) Slotの特徴ベクトル
    
    Returns:
        loss: スカラー
    """
    B, K, D = slot_features.shape
    
    # 正規化
    slot_norm = F.normalize(slot_features, dim=-1)
    
    # 全ペアの類似度
    # (B, K, K)
    similarity = torch.bmm(slot_norm, slot_norm.transpose(1, 2))
    
    # 対角成分を除外
    mask = torch.eye(K, device=slot_features.device).unsqueeze(0).expand(B, -1, -1)
    off_diag = similarity * (1 - mask)
    
    # Contrastive loss: 異なるSlotは類似度を最小化
    loss = off_diag.sum() / (B * K * (K - 1))
    
    return loss


# ハイパーパラメータ推奨値
AGGRESSIVE_CONFIG = {
    'diversity_weight': 5.0,
    'num_slots': 7,
    'lr': 0.0005,
    'num_epochs': 50,
}

VERY_AGGRESSIVE_CONFIG = {
    'diversity_weight': 10.0,
    'num_slots': 10,
    'lr': 0.0003,
    'num_epochs': 100,
}

print("Alternative Diversity Loss Functions Prepared")
print(f"Aggressive config: {AGGRESSIVE_CONFIG}")
print(f"Very aggressive config: {VERY_AGGRESSIVE_CONFIG}")
