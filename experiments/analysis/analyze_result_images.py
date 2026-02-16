"""
Analyze the movi_result.png files to understand what went wrong.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_image(image_path, model_name):
    """Analyze a result visualization image."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Path: {image_path}")
    print(f"{'='*60}")
    
    if not Path(image_path).exists():
        print(f"⚠️  File not found!")
        return None
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"Image shape: {img_array.shape}")
    print(f"Image dtype: {img_array.dtype}")
    
    # Calculate color diversity (as proxy for mask quality)
    # More diverse colors = better mask separation
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    color_ratio = unique_colors / total_pixels
    
    print(f"Unique colors: {unique_colors:,} / {total_pixels:,} ({color_ratio:.2%})")
    
    # Check color variance
    color_std = img_array.std(axis=(0, 1))
    print(f"Color std (RGB): [{color_std[0]:.2f}, {color_std[1]:.2f}, {color_std[2]:.2f}]")
    avg_std = color_std.mean()
    print(f"Average color std: {avg_std:.2f}")
    
    # Diagnosis
    if avg_std < 20:
        print(f"⚠️  WARNING: Very low color variance - possible collapse")
        status = "COLLAPSED"
    elif avg_std < 40:
        print(f"⚠️  WARNING: Low color variance - partial collapse")  
        status = "PARTIAL_COLLAPSE"
    elif color_ratio < 0.01:
        print(f"⚠️  WARNING: Very few unique colors - masks may be collapsed")
        status = "FEW_COLORS"
    else:
        print(f"✓ Color diversity looks reasonable")
        status = "GOOD"
    
    return {
        'model_name': model_name,
        'unique_colors': unique_colors,
        'color_ratio': color_ratio,
        'color_std': color_std,
        'avg_std': avg_std,
        'status': status,
        'image': img_array
    }

def main():
    """Analyze all result images."""
    
    results = []
    
    # Models to analyze
    models = [
        ("checkpoints/temp_scaling_tau05/dinov2_vits14/movi_result.png", 
         "DINOv2 Single Frame τ=0.5 (Baseline)"),
        ("checkpoints/video_stopgrad_200ep/dinov2_vits14/movi_result.png",
         "DINOv2 Video (stopgrad)"),
        ("checkpoints/dinov1_optimized_200ep/dino_vits16/movi_result.png",
         "DINOv1 Video"),
        ("checkpoints/clip_optimized_200ep/clip_vitb16/movi_result.png",
         "CLIP Video"),
    ]
    
    for path, name in models:
        result = analyze_image(path, name)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Avg Std':<10} {'Colors':<10} {'Status':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['model_name']:<40} {r['avg_std']:<10.2f} {r['color_ratio']:<10.2%} {r['status']:<15}")
    
    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if len(results) < 2:
        print("Not enough results to compare")
        return
    
    baseline = results[0]
    video_models = results[1:]
    
    collapsed_count = sum(1 for r in video_models if r['status'] in ['COLLAPSED', 'PARTIAL_COLLAPSE'])
    
    if collapsed_count > 0:
        print(f"\n⚠️  {collapsed_count}/{len(video_models)} video mode models show collapse!")
        print(f"\nBaseline (single frame): Avg std = {baseline['avg_std']:.2f}")
        
        for r in video_models:
            degradation = (r['avg_std'] - baseline['avg_std']) / baseline['avg_std'] * 100
            print(f"  {r['model_name']}: Avg std = {r['avg_std']:.2f} ({degradation:+.1f}%)")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. ❌ Video mode is not working - revert to single-frame mode")
        print(f"  2. ✅ Run single-frame training with τ=0.5 for all backbones:")
        print(f"     - DINOv2 (baseline): Already good")
        print(f"     - DINOv1: Re-train with max_frames=1")
        print(f"     - CLIP: Re-train with max_frames=1")
        print(f"  3. 📊 Compare backbones on single-frame performance for TA")
        print(f"  4. 📝 Document that video mode needs more work (future research)")
    else:
        print(f"✓ All models look reasonable - video mode may be working")

if __name__ == "__main__":
    main()
