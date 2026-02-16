"""
Video Mode Ablation Comparison
================================
3つの設定を比較して、どの崩壊防止策が最も効果的かを分析
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
dataset = MoviDataset('../data/movi_a_subset', split='metal', max_frames=24)
sample = dataset[0]
video = sample['video'].unsqueeze(0).to(device)
seg_gt = sample['segmentation']

print("="*60)
print("Video Mode Ablation Comparison")
print("="*60)
print(f"Sample: {dataset.files[0]}")
print(f"Frames: {video.shape[1]}")
print("")

# Experiment configurations
configs = [
    {
        'name': 'Baseline',
        'checkpoint': '../checkpoints/video_mode_8frames/dinov2_vits14/best_model.pt',
        'refresh_interval': 0,
        'use_stop_gradient': False,
        'description': 'Original (collapses)'
    },
    {
        'name': 'Stop-gradient',
        'checkpoint': '../checkpoints/video_stopgrad/dinov2_vits14/best_model.pt',
        'refresh_interval': 0,
        'use_stop_gradient': True,
        'description': 'Prevents gradient flow through predictor'
    },
    {
        'name': 'Stop-grad + Refresh',
        'checkpoint': '../checkpoints/video_stopgrad_refresh4/dinov2_vits14/best_model.pt',
        'refresh_interval': 4,
        'use_stop_gradient': True,
        'description': 'Combined approach (recommended)'
    }
]

results = {}

for config in configs:
    name = config['name']
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    # Load model
    model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64, mask_temperature=0.5)
    
    try:
        ckpt = torch.load(config['checkpoint'], weights_only=False)
        # Load with strict=False to allow missing keys (e.g., output_norm for old checkpoints)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"⚠ Missing keys (using default initialization): {len(missing)} keys")
        print(f"✓ Loaded checkpoint: {config['checkpoint']}")
    except FileNotFoundError:
        print(f"⚠ Checkpoint not found (will use untrained model): {config['checkpoint']}")
    
    model = model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        result = model.forward_video(
            video,
            return_all_masks=True,
            refresh_interval=config['refresh_interval'],
            use_stop_gradient=config['use_stop_gradient']
        )
    
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    all_slots = result['all_slots'][0]  # (T, K, D)
    t, k = all_slots.shape[:2]
    
    # Metrics
    metrics = {
        'loss': result['total_loss'].item(),
        'slot_diversity_per_frame': [],
        'max_activations_per_frame': [],
        'mask_sim_per_frame': [],
        'slot_assignment_consistency': []
    }
    
    # Per-frame analysis
    for frame_idx in range(t):
        slots_f = all_slots[frame_idx]
        masks_f = all_masks[frame_idx, :, 0]  # (K, 16, 16)
        
        # Slot diversity (lower = more diverse)
        slots_norm = F.normalize(slots_f, dim=-1)
        sim_matrix = torch.mm(slots_norm, slots_norm.t())
        off_diag = (sim_matrix.sum() - sim_matrix.trace()) / (k * (k - 1))
        metrics['slot_diversity_per_frame'].append(off_diag.item())
        
        # Max activations
        max_vals = masks_f.max(dim=-1)[0].max(dim=-1)[0]
        metrics['max_activations_per_frame'].append(max_vals.cpu().numpy())
        
        # Mask similarity (lower = more diverse)
        masks_flat = masks_f.view(k, -1)
        mask_sim = torch.mm(masks_flat, masks_flat.t())
        mask_off_diag = (mask_sim.sum() - mask_sim.trace()) / (k * (k - 1))
        metrics['mask_sim_per_frame'].append(mask_off_diag.item())
    
    # Slot assignment consistency across frames
    masks_resized = F.interpolate(
        all_masks.view(t * k, 1, 16, 16),
        size=seg_gt.shape[-2:],
        mode='bilinear'
    ).view(t, k, seg_gt.shape[-2], seg_gt.shape[-1])
    
    num_objects = seg_gt.max().item() + 1
    for slot_idx in range(k):
        assignments = []
        for frame_idx in range(t):
            seg_frame = seg_gt[frame_idx]
            slot_mask = (masks_resized[frame_idx, slot_idx] > 0.3).float().cpu()
            
            best_iou = 0
            best_obj = -1
            for obj_idx in range(num_objects):
                obj_mask = (seg_frame == obj_idx).float().cpu()
                intersection = (slot_mask * obj_mask).sum().item()
                union = ((slot_mask + obj_mask) > 0).float().sum().item()
                iou = intersection / (union + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj_idx
            assignments.append(best_obj)
        
        # Consistency = most common assignment frequency
        unique = set(assignments)
        consistency = max(assignments.count(obj) for obj in unique) / t * 100
        metrics['slot_assignment_consistency'].append(consistency)
    
    results[name] = metrics
    
    # Print summary
    print(f"\nMetrics Summary:")
    print(f"  Loss: {metrics['loss']:.6f}")
    print(f"  Slot diversity (avg): {np.mean(metrics['slot_diversity_per_frame']):.4f}")
    print(f"  Mask similarity (avg): {np.mean(metrics['mask_sim_per_frame']):.4f}")
    print(f"  Assignment consistency (avg): {np.mean(metrics['slot_assignment_consistency']):.1f}%")
    print(f"  Max activations (final frame): {metrics['max_activations_per_frame'][-1]}")


# Visualization
print(f"\n{'='*60}")
print("Creating comparison plots...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Slot diversity over time
ax = axes[0, 0]
for name, metrics in results.items():
    ax.plot(metrics['slot_diversity_per_frame'], label=name, marker='o')
ax.set_xlabel('Frame')
ax.set_ylabel('Slot Similarity (lower = better)')
ax.set_title('Slot Diversity Across Frames')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mask similarity over time
ax = axes[0, 1]
for name, metrics in results.items():
    ax.plot(metrics['mask_sim_per_frame'], label=name, marker='s')
ax.set_xlabel('Frame')
ax.set_ylabel('Mask Similarity (lower = better)')
ax.set_title('Mask Diversity Across Frames')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Assignment consistency
ax = axes[1, 0]
names = list(results.keys())
consistencies = [np.mean(results[name]['slot_assignment_consistency']) for name in names]
bars = ax.bar(names, consistencies)
for bar, cons in zip(bars, consistencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cons:.1f}%', ha='center', va='bottom')
ax.set_ylabel('Consistency (%)')
ax.set_title('Slot-Object Assignment Consistency')
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Max activations heatmap
ax = axes[1, 1]
for i, name in enumerate(names):
    max_acts = np.array(results[name]['max_activations_per_frame'])  # (T, K)
    t, k = max_acts.shape
    im = ax.imshow(max_acts.T, aspect='auto', cmap='viridis', 
                   extent=[0, t, (i+1)*k, i*k], vmin=0.2, vmax=0.8)
    ax.text(-0.5, i*k + k/2, name, ha='right', va='center', fontsize=10)
ax.set_xlabel('Frame')
ax.set_ylabel('Slot Index (grouped by config)')
ax.set_title('Max Activations per Slot (color = activation strength)')
plt.colorbar(im, ax=ax, label='Max Activation')
ax.set_yticks([])

plt.tight_layout()
save_path = '../checkpoints/video_ablation_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {save_path}")

# Summary table
print(f"\n{'='*60}")
print("SUMMARY TABLE")
print("="*60)
print(f"{'Config':<25} {'Loss':<10} {'Slot Div':<12} {'Mask Div':<12} {'Consistency':<12}")
print("-"*60)
for name, metrics in results.items():
    print(f"{name:<25} "
          f"{metrics['loss']:<10.4f} "
          f"{np.mean(metrics['slot_diversity_per_frame']):<12.4f} "
          f"{np.mean(metrics['mask_sim_per_frame']):<12.4f} "
          f"{np.mean(metrics['slot_assignment_consistency']):<12.1f}%")

print(f"\n{'='*60}")
print("RECOMMENDATIONS")
print("="*60)
print("""
Based on the analysis:

1. **Baseline**: Likely shows increasing slot similarity → collapse
2. **Stop-gradient**: Prevents gradient-based collapse but may still drift
3. **Stop-grad + Refresh**: Best balance - maintains diversity + stability

Expected improvements:
- Slot diversity: Should remain < 0.2 (vs baseline → 0.33)
- Assignment consistency: Should be > 80% (vs baseline ~ 60%)
- Max activations: Should stay diverse (vs baseline → uniform 0.33)

Next steps:
  1. If not trained yet: bash src/train_video_ablations.sh
  2. Review full training logs in logs/video_*.log
  3. Visualize results with checkpoints/*/movi_result.png
""")
