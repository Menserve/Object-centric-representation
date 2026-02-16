"""
実際のフォワードパスを実行してAttention mapとSlot類似度を確認
"""
import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset
import numpy as np
import matplotlib.pyplot as plt

# モデル読み込み
checkpoint_path = '../checkpoints/temp_scaling_tau02/dinov2_vits14/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# データ読み込み
dataset = MoviDataset('../data/movi_a_subset', split='all', max_frames=1)
sample = dataset[0]
video = sample['video'].unsqueeze(0)  # (1, T, 3, H, W)

print("="*80)
print("FULL FORWARD PASS - GROUND TRUTH CHECK")
print("="*80)

with torch.no_grad():
    # フル推論
    recon_feat, target_feat, masks, slots = model.forward_image(video[:, 0])
    
    print(f"\n🔍 Model output:")
    print(f"  Reconstruction: {recon_feat.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Slots: {slots.shape}")
    
    # スロット間の類似度（最終出力）
    slots_np = slots[0].cpu().numpy()  # (5, 64)
    slot_similarities = []
    for i in range(5):
        for j in range(i + 1, 5):
            vec_i = slots_np[i]
            vec_j = slots_np[j]
            sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
            slot_similarities.append(sim)
    
    avg_slot_sim = np.mean(slot_similarities)
    max_slot_sim = np.max(slot_similarities)
    min_slot_sim = np.min(slot_similarities)
    
    print(f"\n📊 Slot vector similarity (final output):")
    print(f"  Mean: {avg_slot_sim:.6f}")
    print(f"  Max: {max_slot_sim:.6f}")
    print(f"  Min: {min_slot_sim:.6f}")
    
    if avg_slot_sim > 0.95:
        print("  ❌ CRITICAL: All slots are nearly identical!")
    elif avg_slot_sim > 0.8:
        print("  ⚠️  WARNING: Slots are very similar")
    else:
        print("  ✅ OK: Slots have diversity")
    
    # マスクの分析
    masks_np = masks[0].cpu().numpy()  # (5, 1, 16, 16)
    masks_flat = masks_np.reshape(5, -1)  # (5, 256)
    
    print(f"\n📊 Mask analysis:")
    for i in range(5):
        mask_i = masks_flat[i]
        coverage = (mask_i > 0.1).sum() / len(mask_i)
        max_val = mask_i.max()
        entropy = -np.sum(mask_i * np.log(mask_i + 1e-8))
        print(f"  Slot {i}: coverage={coverage:.1%}, max={max_val:.4f}, entropy={entropy:.2f}")
    
    # マスク間の類似度
    mask_similarities = []
    for i in range(5):
        for j in range(i + 1, 5):
            mask_i = masks_flat[i]
            mask_j = masks_flat[j]
            sim = np.dot(mask_i, mask_j) / (np.linalg.norm(mask_i) * np.linalg.norm(mask_j) + 1e-8)
            mask_similarities.append(sim)
    
    avg_mask_sim = np.mean(mask_similarities)
    max_mask_sim = np.max(mask_similarities)
    min_mask_sim = np.min(mask_similarities)
    
    print(f"\n📊 Mask similarity (attention patterns):")
    print(f"  Mean: {avg_mask_sim:.6f}")
    print(f"  Max: {max_mask_sim:.6f}")
    print(f"  Min: {min_mask_sim:.6f}")
    
    if avg_mask_sim > 0.95:
        print("  ❌ CRITICAL: All masks are nearly identical!")
        print("     This is complete collapse - all slots attend to everything equally.")
    elif avg_mask_sim > 0.8:
        print("  ⚠️  WARNING: Masks are very similar")
    else:
        print("  ✅ OK: Masks have diversity")
    
    # 可視化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 入力画像
    img = video[0, 0].permute(1, 2, 0).cpu().numpy()
    
    for i in range(5):
        # マスク
        mask_img = masks_np[i, 0]
        axes[0, i].imshow(mask_img, cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Slot {i} Mask\nmax={masks_flat[i].max():.3f}')
        axes[0, i].axis('off')
        
        # スロットベクトルのヒートマップ
        slot_vec = slots_np[i].reshape(8, 8)  # 64次元 → 8×8
        axes[1, i].imshow(slot_vec, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1, i].set_title(f'Slot {i} Vector\nstd={slots_np[i].std():.3f}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Ground Truth Check\nSlot sim: {avg_slot_sim:.4f}, Mask sim: {avg_mask_sim:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('../checkpoints/twolayer_mlp_200ep/dinov2_vits14/ground_truth_check.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: ground_truth_check.png")
    
    # 結論
    print("\n" + "="*80)
    print("GROUND TRUTH DIAGNOSIS:")
    print("="*80)
    
    if avg_slot_sim > 0.9 and avg_mask_sim > 0.9:
        print("❌ COMPLETE COLLAPSE CONFIRMED")
        print("   Both slots and masks are nearly identical.")
        print("   The model has converged to a degenerate solution.")
        print("\n   Possible causes:")
        print("   1. Learning rate too high → uniform averaging")
        print("   2. Diversity loss too weak")
        print("   3. Feature projection collapsing spatial info")
        print("   4. Decoder overpowering slot attention")
    elif avg_mask_sim > 0.8:
        print("⚠️  HIGH MASK SIMILARITY")
        print("   Attention patterns are too uniform.")
    else:
        print("✅ NO COLLAPSE DETECTED")
        print("   Slots and masks show reasonable diversity.")
