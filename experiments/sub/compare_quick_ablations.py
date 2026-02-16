"""
Quick Ablation Results Comparison
==================================
20 epoch結果を比較
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np

log_dir = '../logs'
configs = [
    ('Baseline', 'quick_baseline.log'),
    ('Stop-gradient', 'quick_stopgrad.log'),
    ('Stop-grad + Refresh', 'quick_stopgrad_refresh4.log')
]

print("="*60)
print("Quick Ablation Results (20 epochs)")
print("="*60)

results = {}

for name, log_file in configs:
    path = os.path.join(log_dir, log_file)
    
    if not os.path.exists(path):
        print(f"\n❌ {name}: Log not found - {log_file}")
        continue
    
    with open(path, 'r') as f:
        content = f.read()
    
    # Extract best loss
    best_match = re.search(r'Best loss: ([\d.]+)', content)
    best_loss = float(best_match.group(1)) if best_match else None
    
    # Extract epoch losses
    epoch_losses = []
    for match in re.finditer(r'Epoch \d+/\d+ \| Loss: ([\d.]+)', content):
        epoch_losses.append(float(match.group(1)))
    
    # Extract final metrics from last few lines
    lines = content.split('\n')
    
    results[name] = {
        'best_loss': best_loss,
        'epoch_losses': epoch_losses,
        'log_path': path
    }
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.6f}" if best_loss else "Best loss: N/A")
    print(f"Epochs completed: {len(epoch_losses)}")
    if len(epoch_losses) >= 3:
        print(f"Loss trajectory: {epoch_losses[0]:.3f} → {epoch_losses[len(epoch_losses)//2]:.3f} → {epoch_losses[-1]:.3f}")

# Visualization
if results:
    print(f"\n{'='*60}")
    print("Creating comparison plot...")
    print(f"{'='*60}")
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        if data['epoch_losses']:
            plt.plot(data['epoch_losses'], label=name, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (20 epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final comparison
    plt.subplot(1, 2, 2)
    names = [n for n, d in results.items() if d['best_loss'] is not None]
    losses = [results[n]['best_loss'] for n in names]
    bars = plt.bar(names, losses)
    plt.ylabel('Best Loss')
    plt.title('Final Performance')
    plt.xticks(rotation=15, ha='right')
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = '../checkpoints/quick_ablation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

print(f"\n{'='*60}")
print("INTERPRETATION GUIDE")
print(f"{'='*60}")
print("""
Expected results after 20 epochs:

✅ **Success indicators**:
   - Stop-grad + Refresh: Best loss (lowest)
   - Smooth convergence without oscillations
   - Loss gap between baseline and prevention methods

⚠️ **If all losses are similar**:
   - 20 epochs might be too short to see collapse
   - Check mask analysis with analyze_video_masks.py

❌ **If prevention methods are worse**:
   - Refresh interval might be too aggressive
   - Try refresh_interval=8 instead of 4

Next steps:
  1. Detailed analysis: python src/analyze_video_masks.py
  2. Visual inspection: checkpoints/quick_*/movi_result.png
  3. If promising → Full 200-epoch training
""")

print(f"\n{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")

if results:
    best_name = min(results.items(), key=lambda x: x[1]['best_loss'] or float('inf'))[0]
    print(f"\n🏆 Best configuration: {best_name}")
    print(f"\nTo train full 200 epochs with best config:")
    
    if best_name == 'Stop-grad + Refresh':
        print("\n  python src/train_movi.py \\")
        print("    --backbone dinov2_vits14 \\")
        print("    --num_epochs 200 \\")
        print("    --max_frames 8 \\")
        print("    --use_stop_gradient \\")
        print("    --refresh_interval 4 \\")
        print("    --mask_temperature 0.5 \\")
        print("    --save_dir checkpoints/video_final")
