"""
Quick Ablation Detailed Analysis
=================================
各checkpointのマスク品質を詳細分析
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
dataset = MoviDataset('data/movi_a_subset', split='metal', max_frames=8)
sample = dataset[0]
video = sample['video'].unsqueeze(0).to(device)

print("="*60)
print("Quick Ablation Detailed Analysis (8 frames)")
print("="*60)

configs = [
    {'name': 'Baseline', 'checkpoint': 'checkpoints/quick_baseline/dinov2_vits14/best_model.pt',
     'refresh_interval': 0, 'use_stop_gradient': False},
    {'name': 'Stop-gradient', 'checkpoint': 'checkpoints/quick_stopgrad/dinov2_vits14/best_model.pt',
     'refresh_interval': 0, 'use_stop_gradient': True},
    {'name': 'Stop-grad + Refresh', 'checkpoint': 'checkpoints/quick_stopgrad_refresh4/dinov2_vits14/best_model.pt',
     'refresh_interval': 4, 'use_stop_gradient': True},
]

results = {}

for config in configs:
    name = config['name']
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    # Load model
    model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64, mask_temperature=0.5)
    
    try:
        ckpt = torch.load(config['checkpoint'], weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"✓ Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {config['checkpoint']}")
        continue
    
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
    
    all_slots = result['all_slots'][0]  # (T, K, D)
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    t, k = all_slots.shape[:2]
    
    # Metrics per frame
    slot_divs = []
    max_acts = []
    
    for frame_idx in range(t):
        slots_f = all_slots[frame_idx]
        masks_f = all_masks[frame_idx, :, 0]
        
        # Slot diversity
        slots_norm = F.normalize(slots_f, dim=-1)
        sim_matrix = torch.mm(slots_norm, slots_norm.t())
        off_diag = (sim_matrix.sum() - sim_matrix.trace()) / (k * (k - 1))
        slot_divs.append(off_diag.item())
        
        # Max activations
        max_vals = masks_f.max(dim=-1)[0].max(dim=-1)[0].cpu().numpy()
        max_acts.append(max_vals)
    
    results[name] = {
        'slot_diversity': slot_divs,
        'max_activations': max_acts
    }
    
    # Print summary
    print(f"\nSlot diversity per frame:")
    for i, div in enumerate(slot_divs):
        print(f"  Frame {i}: {div:.4f}")
    
    print(f"\nMax activations (final frame): {max_acts[-1]}")
    print(f"Slot diversity (avg): {np.mean(slot_divs):.4f}")
    print(f"Slot diversity (final): {slot_divs[-1]:.4f}")

# Visualization
print(f"\n{'='*60}")
print("Creating analysis plots...")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Slot diversity trajectory
ax = axes[0]
for name, data in results.items():
    ax.plot(data['slot_diversity'], label=name, marker='o', linewidth=2)
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Target (< 0.2)')
ax.axhline(y=0.33, color='orange', linestyle='--', alpha=0.5, label='Collapse (> 0.33)')
ax.set_xlabel('Frame')
ax.set_ylabel('Slot Similarity (lower = better)')
ax.set_title('Slot Diversity Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Max activations heatmap for last frame
ax = axes[1]
names = list(results.keys())
data_matrix = np.array([results[name]['max_activations'][-1] for name in names])
im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0.2, vmax=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Slot Index')
ax.set_title('Max Activations (Final Frame)')
plt.colorbar(im, ax=ax, label='Activation')
for i in range(len(names)):
    for j in range(5):
        text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
save_path = 'checkpoints/quick_ablation_detailed.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {save_path}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

print(f"\n{'Config':<25} {'Diversity (final)':<20} {'Diversity (avg)':<20}")
print("-"*65)
for name, data in results.items():
    print(f"{name:<25} {data['slot_diversity'][-1]:<20.4f} {np.mean(data['slot_diversity']):<20.4f}")

print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

best_name = min(results.items(), key=lambda x: x[1]['slot_diversity'][-1])[0]
worst_name = max(results.items(), key=lambda x: x[1]['slot_diversity'][-1])[0]

print(f"""
✅ Best (most diverse slots): {best_name}
❌ Worst (most collapsed): {worst_name}

Key findings:
- Diversity < 0.2: Excellent (slots are distinct)
- Diversity 0.2-0.3: Acceptable (moderate diversity)
- Diversity > 0.33: Collapsed (approaching uniform 1/3)

Max activation pattern:
- Diverse: [0.4, 0.5, 0.6, 0.4, 0.5] (varied)
- Collapsed: [0.33, 0.33, 0.33, 0.33, 0.33] (uniform)
- Dominating: [0.2, 0.9, 0.2, 0.2, 0.2] (one slot takes all)

Recommendation:
{f"→ {best_name} shows better diversity control" if best_name != 'Baseline' else "→ All methods are similar at 20 epochs - need longer training"}
{f"→ Consider full 200-epoch training with {best_name}" if best_name != 'Baseline' else "→ Extend to 50-200 epochs to see clearer differences"}
""")

print(f"\nNext steps:")
if any(np.mean(data['slot_diversity']) < 0.25 for data in results.values()):
    best = min(results.items(), key=lambda x: np.mean(x[1]['slot_diversity']))[0]
    print(f"  1. Full training (200 epochs) with {best}")
    if best == 'Stop-grad + Refresh':
        print(f"     python src/train_movi.py --backbone dinov2_vits14 \\")
        print(f"       --num_epochs 200 --max_frames 8 \\")
        print(f"       --use_stop_gradient --refresh_interval 4 \\")
        print(f"       --mask_temperature 0.5 --save_dir checkpoints/video_final")
    elif best == 'Stop-gradient':
        print(f"     python src/train_movi.py --backbone dinov2_vits14 \\")
        print(f"       --num_epochs 200 --max_frames 8 \\")
        print(f"       --use_stop_gradient --mask_temperature 0.5 \\")
        print(f"       --save_dir checkpoints/video_final")
else:
    print(f"  1. All methods similar - try 50-100 epochs first")
    print(f"  2. Or pivot to single-frame + backbone comparison")
