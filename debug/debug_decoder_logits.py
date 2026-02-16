"""
Decoder出力のmask_logitsを確認
"""
import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset
import numpy as np

# モデル読み込み
checkpoint_path = '../checkpoints/twolayer_mlp_200ep/dinov2_vits14/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# データ読み込み
dataset = MoviDataset('../data/movi_a_subset', split='all', max_frames=1)
sample = dataset[0]
video = sample['video'].unsqueeze(0)  # (1, T, 3, H, W)

print("="*80)
print("DECODER MASK LOGITS ANALYSIS")
print("="*80)

with torch.no_grad():
    # フル推論でスロットを取得
    features_proj, _ = model.encode(video[:, 0])
    slots = model.slot_attention(features_proj)  # (1, 5, 64)
    
    print(f"\n🔍 Slots after Slot Attention:")
    print(f"  Shape: {slots.shape}")
    print(f"  Mean: {slots.mean().item():.6f}")
    print(f"  Std: {slots.std().item():.6f}")
    
    # スロットを384次元に変換してDecoderへ
    slots_upsampled = model.slot_to_feature(slots)  # (1, 5, 384)
    
    print(f"\n🔍 Slots after upsampling to 384:")
    print(f"  Shape: {slots_upsampled.shape}")
    print(f"  Mean: {slots_upsampled.mean().item():.6f}")
    print(f"  Std: {slots_upsampled.std().item():.6f}")
   
    # スロット間の類似度
    slots_up_np = slots_upsampled[0].cpu().numpy()  # (5, 384)
    slot_up_similarities = []
    for i in range(5):
        for j in range(i + 1, 5):
            vec_i = slots_up_np[i]
            vec_j = slots_up_np[j]
            sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
            slot_up_similarities.append(sim)
    
    print(f"\n📊 Upsampled slot similarity:")
    print(f"  Mean: {np.mean(slot_up_similarities):.6f}")
    print(f"  Max: {np.max(slot_up_similarities):.6f}")
    print(f"  Min: {np.min(slot_up_similarities):.6f}")
    
    # Decoderの手動実行
    b, k, d = slots_upsampled.shape
    h, w = model.decoder.resolution
    
    # Spatial Broadcast
    slots_2d = slots_upsampled.view(b * k, d, 1, 1).expand(-1, -1, h, w)
    grid = model.decoder.build_grid(b * k, slots_upsampled.device)
    
    decode_in = torch.cat([slots_2d, grid], dim=1)  # (5, 386, 16, 16)
    
    print(f"\n🔍 Decoder input (slot + grid):")
    print(f"  Shape: {decode_in.shape}")
    print(f"  Mean: {decode_in.mean().item():.6f}")
    print(f"  Std: {decode_in.std().item():.6f}")
    
    # Decoderを通す
    out = model.decoder.decoder(decode_in)  # (5, 385, 16, 16)
    out = out.view(b, k, d + 1, h, w)  # (1, 5, 385, 16, 16)
    
    pred_feats = out[:, :, :d, :, :]  # (1, 5, 384, 16, 16)
    mask_logits = out[:, :, d:, :, :]  # (1, 5, 1, 16, 16)
    
    print(f"\n🔍 Raw mask logits (before clipping):")
    print(f"  Shape: {mask_logits.shape}")
    print(f"  Mean: {mask_logits.mean().item():.6f}")
    print(f"  Std: {mask_logits.std().item():.6f}")
    print(f"  Min: {mask_logits.min().item():.6f}")
    print(f"  Max: {mask_logits.max().item():.6f}")
    
    # 各スロットのlogits統計
    mask_logits_np = mask_logits[0, :, 0].cpu().numpy()  # (5, 16, 16)
    print(f"\n📊 Per-slot mask logits:")
    for i in range(5):
        logits_i = mask_logits_np[i].flatten()
        print(f"  Slot {i}: mean={logits_i.mean():.4f}, std={logits_i.std():.4f}, min={logits_i.min():.4f}, max={logits_i.max():.4f}")
    
    # スロット間のlogits類似度
    logits_flat = mask_logits_np.reshape(5, -1)  # (5, 256)
    logit_similarities = []
    for i in range(5):
        for j in range(i + 1, 5):
            logit_i = logits_flat[i]
            logit_j = logits_flat[j]
            # Pearson correlation
            corr = np.corrcoef(logit_i, logit_j)[0, 1]
            logit_similarities.append(corr)
    
    print(f"\n📊 Mask logits correlation (before softmax):")
    print(f"  Mean: {np.mean(logit_similarities):.6f}")
    print(f"  Max: {np.max(logit_similarities):.6f}")
    print(f"  Min: {np.min(logit_similarities):.6f}")
    
    if np.mean(logit_similarities) > 0.95:
        print("\n❌ CRITICAL: Decoder is producing nearly identical logits for all slots!")
        print("   The decoder is NOT differentiating between slots.")
        print("\n   Possible causes:")
        print("   1. Decoder is ignoring slot info, only using coordinate grid")
        print("   2. Upsampled slots (64→384) are too similar")
        print("   3. Decoder capacity insufficient")
    elif np.mean(logit_similarities) > 0.8:
        print("\n⚠️  WARNING: Mask logits are very similar across slots")
    else:
        print("\n✅ OK: Decoder produces diverse logits")
    
    # Softmax後
    mask_logits_clipped = torch.clamp(mask_logits, min=-10, max=10)
    masks = torch.softmax(mask_logits_clipped, dim=1)  # (1, 5, 1, 16, 16)
    
    print(f"\n🔍 After softmax:")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Mean: {masks.mean().item():.6f}")
    print(f"  Std: {masks.std().item():.6f}")
    print(f"  Expected uniform: {1/5:.4f}")
    
    masks_np = masks[0, :, 0].cpu().numpy()
    print(f"\n📊 Per-slot mask coverage (after softmax):")
    for i in range(5):
        mask_i = masks_np[i].flatten()
        coverage = (mask_i > 0.1).sum() / len(mask_i)
        print(f"  Slot {i}: mean={mask_i.mean():.4f}, max={mask_i.max():.4f}, coverage>{0.1:.1f}={coverage:.1%}")
