"""
Diagnose why video mode training failed to produce good masks.
Compare with the successful single-frame + temperature scaling result.
"""

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from savi_dinosaur import SAViDinosaur

def load_and_test_model(checkpoint_path, model_name, max_frames=1):
    """Load a checkpoint and test on a sample."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Determine backbone from path
    if "dino_vits16" in str(checkpoint_path):
        backbone = "dino_vits16"
    elif "clip_vitb16" in str(checkpoint_path):
        backbone = "clip_vitb16"
    else:
        backbone = "dinov2_vits14"
    
    # Load model
    model = SAViDinosaur(
        num_slots=5,
        backbone=backbone,
        slot_dim=64
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a test sample
    data_dir = Path("data/movi_a_subset")
    sample_file = list(data_dir.glob("metal_*.pt"))[0]
    video_data = torch.load(sample_file)
    frames = video_data['video'][:max_frames]  # (T, 3, H, W)
    
    with torch.no_grad():
        if max_frames == 1:
            # Single frame mode
            recon, target, masks, slots = model.forward_image(frames[0:1])
        else:
            # Video mode
            recon, target, masks, slots = model.forward_video_batch(
                frames.unsqueeze(0)  # (1, T, 3, H, W)
            )
            masks = masks[0, -1]  # Last frame: (K, 1, H, W)
            recon = recon[0, -1]  # (3, H, W)
            target = target[0, -1]  # (3, H, W)
    
    # Analyze mask quality
    masks_np = masks.squeeze(1).numpy()  # (K, H, W)
    
    # Calculate metrics
    mask_entropy = -np.sum(masks_np * np.log(masks_np + 1e-8), axis=0).mean()
    mask_max = masks_np.max(axis=0)
    mask_confidence = mask_max.mean()
    
    # Check for collapse
    mask_sums = masks_np.sum(axis=(1, 2))
    mask_variance = mask_sums.var()
    
    # Slot-to-slot similarity
    masks_flat = masks_np.reshape(masks_np.shape[0], -1)
    similarity_matrix = np.corrcoef(masks_flat)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarity = np.abs(similarity_matrix).mean()
    
    print(f"Backbone: {backbone}")
    print(f"Max frames: {max_frames}")
    print(f"Mask entropy: {mask_entropy:.4f} (higher = more diverse)")
    print(f"Mask confidence: {mask_confidence:.4f} (how confident masks are)")
    print(f"Mask sum variance: {mask_variance:.4f} (higher = more diverse usage)")
    print(f"Avg slot similarity: {avg_similarity:.4f} (lower = less collapse)")
    print(f"Mask sums per slot: {mask_sums}")
    
    # Check if collapsed (all masks similar)
    if avg_similarity > 0.8:
        print(f"⚠️  WARNING: Severe slot collapse detected!")
    elif avg_similarity > 0.6:
        print(f"⚠️  WARNING: Moderate slot collapse detected!")
    elif mask_variance < 100:
        print(f"⚠️  WARNING: Low mask diversity (variance: {mask_variance:.2f})")
    else:
        print(f"✓ Masks appear to be reasonably diverse")
    
    return {
        'model_name': model_name,
        'backbone': backbone,
        'max_frames': max_frames,
        'entropy': mask_entropy,
        'confidence': mask_confidence,
        'variance': mask_variance,
        'similarity': avg_similarity,
        'masks': masks_np,
        'recon': recon.numpy(),
        'target': target.numpy()
    }

def main():
    """Compare different models."""
    results = []
    
    # Test the successful single-frame model
    if Path("checkpoints/temp_scaling_tau05/dinov2_vits14/best_model.pt").exists():
        result = load_and_test_model(
            "checkpoints/temp_scaling_tau05/dinov2_vits14/best_model.pt",
            "Single Frame τ=0.5 (Good)",
            max_frames=1
        )
        results.append(result)
    
    # Test video mode DINOv2
    if Path("checkpoints/video_stopgrad_200ep/dinov2_vits14/best_model.pt").exists():
        result = load_and_test_model(
            "checkpoints/video_stopgrad_200ep/dinov2_vits14/best_model.pt",
            "Video DINOv2 (stopgrad)",
            max_frames=8
        )
        results.append(result)
    
    # Test DINOv1
    if Path("checkpoints/dinov1_optimized_200ep/dino_vits16/best_model.pt").exists():
        result = load_and_test_model(
            "checkpoints/dinov1_optimized_200ep/dino_vits16/best_model.pt",
            "Video DINOv1",
            max_frames=8
        )
        results.append(result)
    
    # Test CLIP
    if Path("checkpoints/clip_optimized_200ep/clip_vitb16/best_model.pt").exists():
        result = load_and_test_model(
            "checkpoints/clip_optimized_200ep/clip_vitb16/best_model.pt",
            "Video CLIP",
            max_frames=8
        )
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Entropy':<10} {'Similarity':<12} {'Variance':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_name']:<30} {r['entropy']:<10.4f} {r['similarity']:<12.4f} {r['variance']:<10.2f}")
    
    # Visualization
    fig, axes = plt.subplots(len(results), 7, figsize=(20, 3*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        target_img = result['target'].transpose(1, 2, 0)
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        axes[i, 0].imshow(target_img)
        axes[i, 0].set_title(f"{result['model_name']}\nOriginal")
        axes[i, 0].axis('off')
        
        # Masks
        for slot_idx in range(5):
            axes[i, slot_idx+1].imshow(result['masks'][slot_idx], cmap='hot', vmin=0, vmax=1)
            axes[i, slot_idx+1].set_title(f"Slot {slot_idx+1}")
            axes[i, slot_idx+1].axis('off')
        
        # Reconstruction
        recon_img = result['recon'].transpose(1, 2, 0)
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
        axes[i, 6].imshow(recon_img)
        axes[i, 6].set_title("Reconstruction")
        axes[i, 6].axis('off')
    
    plt.tight_layout()
    plt.savefig("checkpoints/video_failure_diagnosis.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to checkpoints/video_failure_diagnosis.png")
    
    # Analysis conclusion
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        good = results[0]
        bad = results[1]
        
        print(f"\nComparing '{good['model_name']}' (baseline) vs '{bad['model_name']}':")
        print(f"  Entropy: {good['entropy']:.4f} → {bad['entropy']:.4f} ({bad['entropy']/good['entropy']*100-100:+.1f}%)")
        print(f"  Similarity: {good['similarity']:.4f} → {bad['similarity']:.4f} ({bad['similarity']/good['similarity']*100-100:+.1f}%)")
        print(f"  Variance: {good['variance']:.2f} → {bad['variance']:.2f} ({bad['variance']/good['variance']*100-100:+.1f}%)")
        
        if bad['similarity'] > good['similarity'] * 1.2:
            print(f"\n⚠️  VIDEO MODE DEGRADATION DETECTED:")
            print(f"  - Slot similarity increased by {(bad['similarity']/good['similarity']-1)*100:.1f}%")
            print(f"  - This indicates slot collapse in video mode")
            print(f"\n💡 RECOMMENDATION:")
            print(f"  1. Revert to single-frame mode for backbone comparison")
            print(f"  2. Or: Implement stronger collapse prevention (e.g., slot decorrelation loss)")
            print(f"  3. Or: Reduce Slot Predictor complexity (smaller GRU hidden size)")

if __name__ == "__main__":
    main()
