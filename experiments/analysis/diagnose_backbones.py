#!/usr/bin/env python3
"""
Diagnose backbone feature distributions
Compares DINOv2, DINOv1, and CLIP feature statistics
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from savi_dinosaur import FeatureExtractor

def analyze_backbone(backbone_name):
    """Analyze feature statistics for a given backbone"""
    print(f"\n{'='*60}")
    print(f"Analyzing {backbone_name}")
    print(f"{'='*60}")
    
    # Load data
    data_dir = Path('data/movi_a_subset')
    sample_files = sorted(data_dir.glob('*.pt'))[:5]  # Use first 5 samples
    
    # Create feature extractor
    extractor = FeatureExtractor(backbone=backbone_name).cuda().eval()
    
    all_features = []
    
    for sample_file in sample_files:
        data = torch.load(sample_file, weights_only=True)
        images = data['video'][:1]  # Use first frame (T, C, H, W)
        images = images.cuda()  # (1, 3, 256, 256)
        # Resize to 224x224
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        with torch.no_grad():
            features = extractor(images)  # (1, 384, 16, 16)
        
        all_features.append(features.cpu())
    
    # Stack all features
    all_features = torch.cat(all_features, dim=0)  # (5, 384, 16, 16)
    
    # Compute statistics
    print(f"\nFeature shape: {all_features.shape}")
    print(f"Mean: {all_features.mean().item():.6f}")
    print(f"Std: {all_features.std().item():.6f}")
    print(f"Min: {all_features.min().item():.6f}")
    print(f"Max: {all_features.max().item():.6f}")
    
    # Per-channel statistics
    channel_means = all_features.mean(dim=(0, 2, 3))
    channel_stds = all_features.std(dim=(0, 2, 3))
    
    print(f"\nPer-channel stats:")
    print(f"  Mean of means: {channel_means.mean().item():.6f}")
    print(f"  Std of means: {channel_means.std().item():.6f}")
    print(f"  Mean of stds: {channel_stds.mean().item():.6f}")
    print(f"  Std of stds: {channel_stds.std().item():.6f}")
    
    # Check for dead channels (very low variance)
    dead_channels = (channel_stds < 0.01).sum().item()
    print(f"  Dead channels (std < 0.01): {dead_channels} / {all_features.shape[1]}")
    
    # Spatial statistics
    spatial_means = all_features.mean(dim=(0, 1))  # (16, 16)
    spatial_stds = all_features.std(dim=(0, 1))  # (16, 16)
    
    print(f"\nSpatial stats:")
    print(f"  Mean spatial var: {spatial_means.std().item():.6f}")
    print(f"  Mean spatial std: {spatial_stds.mean().item():.6f}")
    
    return {
        'mean': all_features.mean().item(),
        'std': all_features.std().item(),
        'min': all_features.min().item(),
        'max': all_features.max().item(),
        'dead_channels': dead_channels
    }

if __name__ == '__main__':
    backbones = ['dinov2_vits14', 'dino_vits16', 'clip_vitb16']
    
    results = {}
    for backbone in backbones:
        try:
            results[backbone] = analyze_backbone(backbone)
        except Exception as e:
            print(f"\n❌ Error analyzing {backbone}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"{'Backbone':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Dead Ch.':<10}")
    print("-" * 80)
    for backbone, stats in results.items():
        print(f"{backbone:<20} {stats['mean']:<12.6f} {stats['std']:<12.6f} "
              f"{stats['min']:<12.6f} {stats['max']:<12.6f} {stats['dead_channels']:<10}")
