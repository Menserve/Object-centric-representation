"""
DINOv2特徴量とDecoder CNNの詳細分析
=========================================

仮説：ViTのパッチ境界や画像枠の情報が強調されている

確認項目：
1. DINOv2の16×16パッチそれぞれの統計（境界パッチが特異か？）
2. Decoder CNNの各層での出力（どこで枠が生成されるか？）
3. Mask logitsの空間分布（なぜ極端な負の値になるか？）
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from savi_dinosaur import SAViDinosaur


def analyze_dino_features_per_patch(model, video_path: str, save_dir: str):
    """DINOv2の各パッチの統計を分析"""
    
    # データ読み込み
    data = torch.load(video_path, weights_only=False)
    video = data['video'][:1]
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    img = video[0:1].to(device)
    
    print("="*60)
    print("DINOv2 Features: Per-Patch Analysis")
    print("="*60)
    
    with torch.no_grad():
        features = model.feature_extractor(img)  # (1, 384, 16, 16)
    
    features_np = features[0].cpu().numpy()  # (384, 16, 16)
    h, w = 16, 16
    
    # 各パッチの統計
    print("\nPer-patch statistics (384 dimensions per patch):")
    print("Format: (row, col) -> mean, std, min, max")
    
    # 境界パッチと中心パッチを特定
    border_patches = []
    center_patches = []
    
    patch_stats = np.zeros((h, w, 4))  # mean, std, min, max
    
    for i in range(h):
        for j in range(w):
            patch = features_np[:, i, j]  # (384,)
            patch_mean = patch.mean()
            patch_std = patch.std()
            patch_min = patch.min()
            patch_max = patch.max()
            
            patch_stats[i, j] = [patch_mean, patch_std, patch_min, patch_max]
            
            # 境界判定
            is_border = (i == 0 or i == h-1 or j == 0 or j == w-1)
            if is_border:
                border_patches.append((i, j, patch_mean, patch_std))
            else:
                center_patches.append((i, j, patch_mean, patch_std))
    
    # 境界 vs 中心の統計比較
    border_means = [p[2] for p in border_patches]
    border_stds = [p[3] for p in border_patches]
    center_means = [p[2] for p in center_patches]
    center_stds = [p[3] for p in center_patches]
    
    print(f"\n{'='*60}")
    print("Border vs Center Patch Comparison:")
    print(f"{'='*60}")
    print(f"Border patches (n={len(border_patches)}):")
    print(f"  Mean of means: {np.mean(border_means):.4f}")
    print(f"  Mean of stds:  {np.mean(border_stds):.4f}")
    print(f"  Range: [{np.min(border_means):.4f}, {np.max(border_means):.4f}]")
    
    print(f"\nCenter patches (n={len(center_patches)}):")
    print(f"  Mean of means: {np.mean(center_means):.4f}")
    print(f"  Mean of stds:  {np.mean(center_stds):.4f}")
    print(f"  Range: [{np.min(center_means):.4f}, {np.max(center_means):.4f}]")
    
    # Difference
    mean_diff = abs(np.mean(border_means) - np.mean(center_means))
    print(f"\n⚠️  Difference: {mean_diff:.4f}")
    if mean_diff > 0.5:
        print("  → SIGNIFICANT difference detected! Border patches are distinct.")
    else:
        print("  → Difference is small. Border patches are similar to center.")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Mean map
    im0 = axes[0, 0].imshow(patch_stats[:, :, 0], cmap='RdBu_r')
    axes[0, 0].set_title('Patch Mean (per patch)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Add border indicator
    for i in range(h):
        for j in range(w):
            if i == 0 or i == h-1 or j == 0 or j == w-1:
                axes[0, 0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                   fill=False, edgecolor='red', linewidth=2))
    
    # Std map
    im1 = axes[0, 1].imshow(patch_stats[:, :, 1], cmap='viridis')
    axes[0, 1].set_title('Patch Std (per patch)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Min map
    im2 = axes[1, 0].imshow(patch_stats[:, :, 2], cmap='viridis')
    axes[1, 0].set_title('Patch Min (per patch)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Max map
    im3 = axes[1, 1].imshow(patch_stats[:, :, 3], cmap='viridis')
    axes[1, 1].set_title('Patch Max (per patch)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.suptitle('DINOv2 Feature Statistics per Patch\n(Red border = edge patches)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dino_patch_statistics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_dir}/dino_patch_statistics.png")
    plt.close()


def analyze_decoder_layer_by_layer(model, video_path: str, save_dir: str):
    """Decoder CNNの各層での出力を確認"""
    
    # データ読み込み
    data = torch.load(video_path, weights_only=False)
    video = data['video'][:1]
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    img = video[0:1].to(device)
    
    print("\n" + "="*60)
    print("Decoder CNN: Layer-by-Layer Analysis")
    print("="*60)
    
    with torch.no_grad():
        # Feature extraction
        features_flat, features = model.encode(img)
        slots = model.slot_attention(features_flat, slots_init=None)
        
        # Prepare decoder input
        b, k, d = slots.shape
        h, w = 16, 16
        slots_2d = slots.view(b * k, d, 1, 1).expand(-1, -1, h, w)
        grid = model.decoder.build_grid(b * k, device)
        decode_in = torch.cat([slots_2d, grid], dim=1)
        
        # Hook each layer
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        handles = []
        for i, layer in enumerate(model.decoder.decoder):
            handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
            handles.append(handle)
        
        # Forward pass
        out = model.decoder.decoder(decode_in)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    # Analyze each layer
    print("\nLayer outputs:")
    layer_names = []
    layer_outputs = []
    
    for i, (name, activation) in enumerate(activations.items()):
        layer_names.append(name)
        layer_outputs.append(activation[0].cpu().numpy())  # First slot
        
        print(f"{name}: {activation.shape}")
        print(f"  Range: [{activation.min():.2f}, {activation.max():.2f}]")
        print(f"  Mean: {activation.mean():.4f}, Std: {activation.std():.4f}")
    
    # Visualize first 4 channels of each layer output (for Slot 0)
    num_layers = len(layer_outputs)
    fig, axes = plt.subplots(num_layers, 4, figsize=(12, 3 * num_layers))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, (name, output) in enumerate(zip(layer_names, layer_outputs)):
        for ch_idx in range(min(4, output.shape[0])):
            ax = axes[layer_idx, ch_idx]
            im = ax.imshow(output[ch_idx], cmap='RdBu_r')
            ax.set_title(f'{name} ch{ch_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Decoder CNN: Layer-by-Layer Outputs (Slot 0)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/decoder_layer_by_layer.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_dir}/decoder_layer_by_layer.png")
    plt.close()
    
    # Final mask logits analysis
    out_reshaped = out.view(1, model.num_slots, model.feat_dim + 1, h, w)
    mask_logits = out_reshaped[:, :, model.feat_dim:, :, :]  # (1, K, 1, H, W)
    
    print("\n" + "="*60)
    print("Mask Logits Analysis (before softmax)")
    print("="*60)
    
    for slot_idx in range(model.num_slots):
        logits = mask_logits[0, slot_idx, 0].cpu().numpy()
        print(f"\nSlot {slot_idx}:")
        print(f"  Range: [{logits.min():.2f}, {logits.max():.2f}]")
        print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
        
        # Check if extreme values are at edges
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        
        border_vals = logits[border_mask]
        center_vals = logits[~border_mask]
        
        print(f"  Border: mean={border_vals.mean():.2f}, min={border_vals.min():.2f}, max={border_vals.max():.2f}")
        print(f"  Center: mean={center_vals.mean():.2f}, min={center_vals.min():.2f}, max={center_vals.max():.2f}")
    
    # Visualize mask logits
    fig, axes = plt.subplots(1, model.num_slots, figsize=(3 * model.num_slots, 3))
    
    for slot_idx in range(model.num_slots):
        logits = mask_logits[0, slot_idx, 0].cpu().numpy()
        
        # Use symmetric colormap around 0
        vmax = max(abs(logits.min()), abs(logits.max()))
        vmin = -vmax
        
        im = axes[slot_idx].imshow(logits, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[slot_idx].set_title(f'Slot {slot_idx}\n[{logits.min():.1f}, {logits.max():.1f}]')
        axes[slot_idx].axis('off')
        
        # Add border
        for spine in axes[slot_idx].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        
        plt.colorbar(im, ax=axes[slot_idx], fraction=0.046)
    
    plt.suptitle('Mask Logits (before softmax) - Red border shows spatial extent', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mask_logits_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_dir}/mask_logits_analysis.png")
    plt.close()


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
    
    analyze_dino_features_per_patch(model, video_path, save_dir)
    analyze_decoder_layer_by_layer(model, video_path, save_dir)
    
    print("\n" + "="*60)
    print("✅ Analysis completed!")
    print("="*60)
