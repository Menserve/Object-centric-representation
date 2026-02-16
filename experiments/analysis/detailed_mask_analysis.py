"""詳細なマスク分析と視覚化"""
import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# モデル読み込み
checkpoint_path = '../checkpoints/temp_scaling_tau05/dinov2_vits14/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# データ読み込み（metal sample）
dataset = MoviDataset('../data/movi_a_subset', split='all', max_frames=1)
metal_indices = [i for i, name in enumerate(dataset.files) if 'metal' in name]
sample = dataset[metal_indices[0]]
video = sample['video'].unsqueeze(0)  # (1, T, 3, H, W)

print("="*80)
print("DETAILED MASK ANALYSIS")
print("="*80)

with torch.no_grad():
    # フル推論（forward_imageを使用）
    img_tensor = video[0, 0]  # (3, H, W)
    recon_combined, target, masks, slots = model.forward_image(img_tensor.unsqueeze(0))
    
    # 元画像
    img = video[0, 0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    # マスク
    masks_vis = masks[0, :, 0].numpy()  # (K, H, W)
    
    # スロット類似度
    slots_2d = slots.squeeze(0)  # (K, D)
    slot_corr = torch.corrcoef(slots_2d).numpy()
    
    print("\n1. SLOT CORRELATION MATRIX:")
    print(slot_corr)
    
    print("\n2. MASK STATISTICS:")
    for i in range(5):
        mask = masks_vis[i]
        print(f"\nSlot {i}:")
        print(f"  Mean: {mask.mean():.4f}")
        print(f"  Std:  {mask.std():.4f}")
        print(f"  Min:  {mask.min():.4f}")
        print(f"  Max:  {mask.max():.4f}")
        print(f"  Coverage (>0.5): {(mask > 0.5).mean()*100:.1f}%")
        print(f"  Entropy: {-np.sum(mask * np.log(mask + 1e-8)):.2f}")
        
        # ピーク位置
        peak_y, peak_x = np.unravel_index(mask.argmax(), mask.shape)
        print(f"  Peak location: ({peak_x}, {peak_y})")
    
    print("\n3. MASK PAIRWISE CORRELATION:")
    masks_flat = masks[0, :, 0].view(5, -1)
    mask_corr = torch.corrcoef(masks_flat).numpy()
    print(mask_corr)
    
    # 視覚化
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: 元画像 + 5つのマスク
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.set_title('Original Image', fontsize=10)
    ax0.axis('off')
    
    for i in range(5):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(masks_vis[i], cmap='hot', vmin=0, vmax=1)
        coverage = (masks_vis[i] > 0.5).mean() * 100
        entropy = -np.sum(masks_vis[i] * np.log(masks_vis[i] + 1e-8))
        ax.set_title(f'Slot {i}\nCov: {coverage:.1f}%\nEnt: {entropy:.1f}', 
                     fontsize=9)
        ax.axis('off')
    
    # Row 2: マスク×画像のオーバーレイ
    ax0 = fig.add_subplot(gs[1, 0])
    # 画像をリサイズ（16x16）
    import torch.nn.functional as F
    img_small = F.interpolate(
        torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(),
        size=(16, 16),
        mode='bilinear',
        align_corners=False
    )[0].permute(1, 2, 0).numpy()
    ax0.imshow(img_small)
    ax0.set_title('Reference (16x16)', fontsize=10)
    ax0.axis('off')
    
    for i in range(5):
        ax = fig.add_subplot(gs[1, i+1])
        # マスクをカラーマップで可視化し、画像に重ねる
        overlay = img_small.copy()
        mask_colored = plt.cm.hot(masks_vis[i])[:, :, :3]
        overlay = 0.5 * overlay + 0.5 * mask_colored
        ax.imshow(overlay)
        ax.set_title(f'Slot {i} Overlay', fontsize=9)
        ax.axis('off')
    
    # Row 3: Binary masks (>0.2 threshold to see something)
    ax0 = fig.add_subplot(gs[2, 0])
    ax0.imshow(img_small)
    ax0.set_title('Reference (16x16)', fontsize=10)
    ax0.axis('off')
    
    for i in range(5):
        ax = fig.add_subplot(gs[2, i+1])
        binary_mask = (masks_vis[i] > 0.2).astype(float)  # Lower threshold
        ax.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Slot {i} >0.2', fontsize=9)
        ax.axis('off')
    
    plt.savefig('detailed_mask_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: detailed_mask_analysis.png")
    
    # 追加: Mask logits distribution
    print("\n4. ANALYZING MASK GENERATION...")
    
    # Decoder forward to get mask logits before softmax
    slots_upsampled = model.slot_to_feature(slots)  # (B, K, 384)
    recon_combined, pred_feats, masks_from_decoder = model.decoder(
        slots_upsampled, model.num_slots
    )
    
    # Check decoder output variance
    print("\nDecoder output statistics:")
    print(f"  pred_feats mean: {pred_feats.mean():.4f}, std: {pred_feats.std():.4f}")
    print(f"  pred_feats range: [{pred_feats.min():.4f}, {pred_feats.max():.4f}]")
    
    # Check mask logits (before softmax)
    # Decoderの内部でsoftmaxされているので、逆算はできないが、
    # マスクの sharpness を確認
    print("\nMask sharpness (std of each mask):")
    for i in range(5):
        mask_std = masks_vis[i].std()
        print(f"  Slot {i}: {mask_std:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
