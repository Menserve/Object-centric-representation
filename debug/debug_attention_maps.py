#!/usr/bin/env python3
"""
Slot Attentionの内部Attention mapを可視化
各スロットがどの画像領域に注目しているかを確認
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train_movi import MoviDataset
from savi_dinosaur import SAViDinosaur


def visualize_attention_maps(checkpoint_path: str, data_dir: str, sample_idx: int = 0):
    """
    Slot AttentionのAttention weightを可視化
    
    各スロットが画像のどの領域に注目しているかを確認することで、
    slot collapseの原因を診断する。
    """
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Loaded: Best loss = {checkpoint['loss']:.6f}")
    
    # モデルの構築
    model = SAViDinosaur(backbone='dinov2_vits14', num_slots=5, slot_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # データのロード
    dataset = MoviDataset(data_dir, split='test', max_frames=1)
    sample = dataset[sample_idx]
    
    print(f"\nSample info:")
    print(f"  Materials: {sample['materials']}")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Segmentation shape: {sample['segmentation'].shape}")
    
    # フォワードパス（Attention weightsを取得）
    video = sample['video'].unsqueeze(0)  # (1, T, C, H, W)
    
    with torch.no_grad():
        # 特徴量抽出
        features_proj, features_orig = model.encode(video[:, 0])  # (1, N, 64), (1, 384, 16, 16)
        b, n, d = features_proj.shape
        
        # 特徴量の正規化（Slot Attentionの最初の処理）
        features_proj = model.slot_attention.norm_features(features_proj)
        
        # Slot Attentionの内部処理を再現
        slots = model.slot_attention.slots_mu.expand(b, -1, -1)  # (1, K, 64)
        
        # Key, Valueは全iterationで共通
        k = model.slot_attention.to_k(features_proj)  # (B, N, D)
        v = model.slot_attention.to_v(features_proj)  # (B, N, D)
        
        # Slot Attention iterationごとのAttention mapを記録
        attention_maps_per_iter = []
        
        for iter_idx in range(5):  # 5 iterations
            # Attention計算
            slots_prev = slots
            slots = model.slot_attention.norm_slots(slots)
            
            # Query
            q = model.slot_attention.to_q(slots)  # (B, K, D)
            
            # Attention weights
            scale = d ** -0.5
            dots = torch.einsum('bid,bjd->bij', q, k) * scale  # (B, K, N)
            attn = dots.softmax(dim=1) + 1e-8  # (B, K, N) - 各ピクセルがどのスロットに属するか
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn_normalized = attn / attn_sum
            
            # 記録
            attention_maps_per_iter.append(attn[0].cpu().numpy())  # (K, N)
            
            # Updates
            updates = torch.einsum('bjd,bij->bid', v, attn_normalized)  # (B, K, D)
            
            # GRU
            slots = model.slot_attention.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + model.slot_attention.mlp(model.slot_attention.norm_pre_ff(slots))
    
    # 可視化
    k_slots = model.num_slots
    h, w = 16, 16  # DINOv2のパッチ解像度
    
    fig, axes = plt.subplots(6, k_slots + 2, figsize=(3 * (k_slots + 2), 18))
    
    # 入力画像（最上段共通）
    frame = sample['video'][0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    for col in range(k_slots + 2):
        axes[0, col].imshow(frame)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_title("Input Image", fontsize=10, fontweight='bold')
        elif col == 1:
            axes[0, col].set_title("GT Segmentation", fontsize=10, fontweight='bold')
        else:
            axes[0, col].set_title(f"Slot {col - 2}", fontsize=10, fontweight='bold')
    
    # GT Segmentation（2列目）
    seg = sample['segmentation'][0].cpu().numpy()
    axes[0, 1].clear()
    axes[0, 1].imshow(seg, cmap='tab10')
    axes[0, 1].axis('off')
    
    # 各iteration（1-5）のAttention map
    for iter_idx in range(5):
        row_idx = iter_idx + 1
        attn_map = attention_maps_per_iter[iter_idx]  # (K, N)
        
        # 入力画像（参考用）
        axes[row_idx, 0].imshow(frame)
        axes[row_idx, 0].set_title(f"Iter {iter_idx + 1}", fontsize=9)
        axes[row_idx, 0].axis('off')
        
        # GT（参考用）
        axes[row_idx, 1].imshow(seg, cmap='tab10')
        axes[row_idx, 1].axis('off')
        
        # 各スロットのAttention map
        for slot_idx in range(k_slots):
            attn_slot = attn_map[slot_idx].reshape(h, w)  # (16, 16)
            
            # 統計情報
            attn_mean = attn_slot.mean()
            attn_max = attn_slot.max()
            attn_coverage = (attn_slot > 0.1).sum() / (h * w)  # 10%以上の領域
            
            axes[row_idx, slot_idx + 2].imshow(attn_slot, cmap='hot', vmin=0, vmax=0.5)
            axes[row_idx, slot_idx + 2].set_title(
                f"μ={attn_mean:.3f} max={attn_max:.3f}\ncov={attn_coverage:.1%}",
                fontsize=7
            )
            axes[row_idx, slot_idx + 2].axis('off')
    
    plt.suptitle(
        f"Slot Attention Maps (Materials: {sample['materials']})\n"
        f"Heatmap: どの画像領域に注目しているか（赤=高注目度）",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    # 保存
    save_path = Path(checkpoint_path).parent / 'attention_maps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {save_path}")
    
    # 診断メッセージ
    print("\n" + "=" * 60)
    print("診断結果:")
    print("=" * 60)
    
    final_attn = attention_maps_per_iter[-1]  # (K, N)
    
    for slot_idx in range(k_slots):
        attn_slot = final_attn[slot_idx]
        coverage = (attn_slot > 0.1).sum() / len(attn_slot)
        max_attn = attn_slot.max()
        entropy = -np.sum(attn_slot * np.log(attn_slot + 1e-8))
        
        print(f"Slot {slot_idx}:")
        print(f"  Coverage (>0.1): {coverage:.1%}")
        print(f"  Max attention: {max_attn:.4f}")
        print(f"  Entropy: {entropy:.2f}")
    
    # Slot間の類似度
    print("\nSlot間のAttention pattern類似度（コサイン類似度）:")
    for i in range(k_slots):
        for j in range(i + 1, k_slots):
            attn_i = final_attn[i]
            attn_j = final_attn[j]
            similarity = np.dot(attn_i, attn_j) / (np.linalg.norm(attn_i) * np.linalg.norm(attn_j) + 1e-8)
            print(f"  Slot {i} ↔ Slot {j}: {similarity:.4f}")
    
    print("\n期待される動作:")
    print("  ✅ 各スロットが異なる領域に注目（Attention mapが異なる）")
    print("  ✅ Coverage < 50%（全画像をカバーしない）")
    print("  ✅ 類似度 < 0.5（スロット間で異なるパターン）")
    print("\nSlot Collapseの兆候:")
    print("  ❌ 全スロットが均一に分散（Coverage ≈ 100%）")
    print("  ❌ 類似度 > 0.9（すべてのスロットが同じ領域を見ている）")
    print("  ❌ Entropy低い（一部のピクセルに集中しすぎ）")


if __name__ == '__main__':
    checkpoint_path = '../checkpoints/single_frame_spatial/dinov2_vits14/best_model.pt'
    data_dir = '../data/movi_a_subset'
    
    visualize_attention_maps(checkpoint_path, data_dir, sample_idx=0)
