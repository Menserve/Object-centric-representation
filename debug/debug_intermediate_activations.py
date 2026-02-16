"""
中間表現の徹底的なデバッグ
=================================

各処理ステップでの空間情報の流れを可視化：
1. DINOv2特徴量の抽出（reshape前後）
2. Slot Attentionへの入力
3. Decoderのspatial broadcast
4. Decoder出力（マスク生成前）
5. 最終マスク

「四角い枠」アーティファクトがどこで発生するかを特定
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from savi_dinosaur import SAViDinosaur


def detailed_forward_with_visualization(model, video_path: str, save_dir: str):
    """各段階での中間表現を可視化"""
    
    # データ読み込み
    data = torch.load(video_path, weights_only=False)
    video = data['video'][:1]  # 1フレーム
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    img = video[0:1].to(device)
    
    print("="*60)
    print("Step-by-step Forward Pass Visualization")
    print("="*60)
    
    with torch.no_grad():
        # ===== Step 1: Feature Extraction =====
        print("\n[Step 1] Feature Extraction (FeatureExtractor)")
        features = model.feature_extractor(img)  # (B, D, H, W)
        print(f"  Output shape: {features.shape}")
        print(f"  Expected: (1, 384, 16, 16)")
        print(f"  Range: [{features.min():.2f}, {features.max():.2f}]")
        
        # ===== Step 2: Encode (reshape to sequence) =====
        print("\n[Step 2] Encode: (B, D, H, W) -> (B, N, D)")
        b, c, h, w = features.shape
        features_perm = features.permute(0, 2, 3, 1)  # (B, H, W, D)
        print(f"  After permute(0, 2, 3, 1): {features_perm.shape}")
        print(f"  Expected: (1, 16, 16, 384)")
        
        features_flat = features_perm.reshape(b, -1, c)  # (B, N, D)
        print(f"  After reshape(b, -1, c): {features_flat.shape}")
        print(f"  Expected: (1, 256, 384)")
        print(f"  Range: [{features_flat.min():.2f}, {features_flat.max():.2f}]")
        
        # Test: 逆変換で元に戻るか？
        features_reconstructed = features_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        reconstruction_error = (features - features_reconstructed).abs().max()
        print(f"  Reconstruction error: {reconstruction_error:.6f} (should be 0)")
        if reconstruction_error > 1e-5:
            print("  ⚠️  WARNING: Reshape is NOT reversible!")
        
        # ===== Step 3: Slot Attention =====
        print("\n[Step 3] Slot Attention")
        slots = model.slot_attention(features_flat, slots_init=None)
        print(f"  Output shape: {slots.shape}")
        print(f"  Expected: (1, 5, 384)")
        print(f"  Range: [{slots.min():.2f}, {slots.max():.2f}]")
        
        # Slot diversity
        slots_norm = F.normalize(slots[0], dim=-1)
        similarity = torch.mm(slots_norm, slots_norm.t())
        print(f"  Slot similarity (off-diagonal mean): {(similarity.sum() - similarity.trace()).item() / 20:.4f}")
        
        # ===== Step 4: Decoder - Spatial Broadcast =====
        print("\n[Step 4] Decoder: Spatial Broadcast")
        k = model.num_slots
        d = model.feat_dim
        slots_2d = slots.view(b * k, d, 1, 1).expand(-1, -1, h, w)
        print(f"  Slots after broadcast: {slots_2d.shape}")
        print(f"  Expected: (5, 384, 16, 16)")
        
        grid = model.decoder.build_grid(b * k, device)
        print(f"  Grid shape: {grid.shape}")
        print(f"  Expected: (5, 2, 16, 16)")
        print(f"  Grid X range: [{grid[:, 0].min():.2f}, {grid[:, 0].max():.2f}]")
        print(f"  Grid Y range: [{grid[:, 1].min():.2f}, {grid[:, 1].max():.2f}]")
        
        decode_in = torch.cat([slots_2d, grid], dim=1)
        print(f"  Decoder input (slots + grid): {decode_in.shape}")
        print(f"  Expected: (5, 386, 16, 16)")
        
        # ===== Step 5: Decoder CNN =====
        print("\n[Step 5] Decoder: CNN")
        out = model.decoder.decoder(decode_in)
        print(f"  Decoder output: {out.shape}")
        print(f"  Expected: (5, 385, 16, 16)")
        
        out = out.view(b, k, d + 1, h, w)
        print(f"  After view(b, k, d+1, h, w): {out.shape}")
        print(f"  Expected: (1, 5, 385, 16, 16)")
        
        pred_feats = out[:, :, :d, :, :]
        masks_logits = out[:, :, d:, :, :]
        print(f"  Predicted features: {pred_feats.shape}")
        print(f"  Mask logits (before softmax): {masks_logits.shape}")
        print(f"  Mask logits range: [{masks_logits.min():.2f}, {masks_logits.max():.2f}]")
        
        # ===== Step 6: Softmax =====
        print("\n[Step 6] Mask Softmax")
        masks = torch.softmax(masks_logits, dim=1)
        print(f"  Masks (after softmax): {masks.shape}")
        print(f"  Expected: (1, 5, 1, 16, 16)")
        print(f"  Masks range: [{masks.min():.2f}, {masks.max():.2f}]")
        print(f"  Masks sum along slots: {masks[0].sum(dim=0).mean():.4f} (should be 1.0)")
        
    # ===== Visualization =====
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Figure 1: Feature maps at different stages
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Row 1: Input and first 4 DINOv2 feature channels
    img_np = img[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    features_np = features[0].cpu().numpy()
    for i in range(4):
        im = axes[0, i+1].imshow(features_np[i], cmap='viridis')
        axes[0, i+1].set_title(f'DINO Ch{i}')
        axes[0, i+1].axis('off')
        plt.colorbar(im, ax=axes[0, i+1], fraction=0.046)
    
    # Row 2: Grid visualization
    grid_np = grid[0].cpu().numpy()  # (2, 16, 16)
    axes[1, 0].imshow(grid_np[0], cmap='viridis', vmin=-1, vmax=1)
    axes[1, 0].set_title('Grid X')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(grid_np[1], cmap='viridis', vmin=-1, vmax=1)
    axes[1, 1].set_title('Grid Y')
    axes[1, 1].axis('off')
    
    # Mask logits for first 3 slots
    masks_logits_np = masks_logits[0].cpu().numpy()
    for i in range(3):
        im = axes[1, i+2].imshow(masks_logits_np[i, 0], cmap='RdBu_r')
        axes[1, i+2].set_title(f'Slot {i} logits')
        axes[1, i+2].axis('off')
        plt.colorbar(im, ax=axes[1, i+2], fraction=0.046)
    
    # Row 3: Final masks (all 5 slots)
    masks_np = masks[0].cpu().numpy()
    for i in range(5):
        im = axes[2, i].imshow(masks_np[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[2, i].set_title(f'Slot {i} mask')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
    
    plt.suptitle('Intermediate Activations: Feature Flow', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/intermediate_activations.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {save_dir}/intermediate_activations.png")
    plt.close()
    
    # Figure 2: Detailed mask analysis (high resolution)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Resize masks for visualization
    masks_resized = F.interpolate(
        masks[0],  # (5, 1, 16, 16)
        size=(224, 224),
        mode='bilinear'
    ).squeeze(1).cpu().numpy()  # (5, 224, 224)
    
    for i in range(5):
        # Mask overlay on image
        axes[0, i].imshow(img_np)
        axes[0, i].imshow(masks_resized[i], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f'Slot {i} overlay')
        axes[0, i].axis('off')
        
        # Mask only
        im = axes[1, i].imshow(masks_resized[i], cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Slot {i} mask')
        axes[1, i].axis('off')
        
        # Statistics
        mask_mean = masks_resized[i].mean()
        mask_std = masks_resized[i].std()
        axes[1, i].text(0.5, -0.1, f'μ={mask_mean:.3f}, σ={mask_std:.3f}',
                       transform=axes[1, i].transAxes, ha='center', fontsize=8)
    
    plt.suptitle('Mask Visualization (Resized to 224x224)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/masks_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {save_dir}/masks_detailed.png")
    plt.close()
    
    # Figure 3: Slot assignment (argmax)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # At 16x16 resolution
    argmax_mask = masks[0].argmax(dim=0).squeeze().cpu().numpy()
    im0 = axes[0].imshow(argmax_mask, cmap='tab10', vmin=0, vmax=4)
    axes[0].set_title('Slot Assignment (16x16)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], ticks=range(5))
    
    # At 224x224 resolution
    masks_resized_torch = torch.from_numpy(masks_resized).unsqueeze(1)
    argmax_mask_resized = masks_resized_torch.argmax(dim=0).squeeze().numpy()
    im1 = axes[1].imshow(argmax_mask_resized, cmap='tab10', vmin=0, vmax=4)
    axes[1].set_title('Slot Assignment (224x224)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], ticks=range(5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/slot_assignment_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {save_dir}/slot_assignment_detailed.png")
    plt.close()
    
    # Coverage statistics
    print("\n=== Slot Coverage Statistics ===")
    for i in range(5):
        coverage = (argmax_mask == i).sum() / argmax_mask.size * 100
        print(f"Slot {i}: {coverage:.2f}%")
    
    print("\n" + "="*60)
    print("Visualization completed!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        checkpoint_path = '../checkpoints/symmetry_break_dinov2/dinov2_vits14/best_model.pt'
    else:
        checkpoint_path = sys.argv[1]
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SAViDinosaur(num_slots=5, feat_dim=384, backbone='dinov2_vits14')
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        print("No checkpoint found, using random initialization")
    
    # Test video
    video_path = '../data/movi_a_subset/metal_000.pt'
    save_dir = '../checkpoints/debug_intermediate'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    detailed_forward_with_visualization(model, video_path, save_dir)
