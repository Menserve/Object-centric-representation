"""
Test Slot Predictor Collapse Prevention
========================================
新機能のクイックテスト：
1. refresh_interval パラメータ
2. use_stop_gradient パラメータ
3. SlotPredictor の output normalization
"""

import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Create model
print("1. Creating model...")
model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64, mask_temperature=0.5)
model = model.to(device)
print("✓ Model created\n")

# Create dummy video
print("2. Creating dummy video (B=1, T=8, C=3, H=224, W=224)...")
video = torch.randn(1, 8, 3, 224, 224).to(device)
print("✓ Dummy video created\n")

# Test 1: Baseline (no prevention)
print("3. Test baseline (no prevention)...")
model.eval()
with torch.no_grad():
    result = model.forward_video(video, return_all_masks=True, refresh_interval=0, use_stop_gradient=False)
print(f"✓ Output shapes: masks={result['all_masks'].shape}, slots={result['all_slots'].shape}")
print(f"  Loss: {result['total_loss'].item():.6f}\n")

# Test 2: Stop-gradient
print("4. Test stop-gradient...")
with torch.no_grad():
    result = model.forward_video(video, return_all_masks=True, refresh_interval=0, use_stop_gradient=True)
print(f"✓ Output shapes: masks={result['all_masks'].shape}, slots={result['all_slots'].shape}")
print(f"  Loss: {result['total_loss'].item():.6f}\n")

# Test 3: Refresh interval
print("5. Test refresh_interval=4...")
with torch.no_grad():
    result = model.forward_video(video, return_all_masks=True, refresh_interval=4, use_stop_gradient=False)
print(f"✓ Output shapes: masks={result['all_masks'].shape}, slots={result['all_slots'].shape}")
print(f"  Loss: {result['total_loss'].item():.6f}\n")

# Test 4: Combined (stop-gradient + refresh)
print("6. Test combined (stop-gradient + refresh_interval=4)...")
with torch.no_grad():
    result = model.forward_video(video, return_all_masks=True, refresh_interval=4, use_stop_gradient=True)
print(f"✓ Output shapes: masks={result['all_masks'].shape}, slots={result['all_slots'].shape}")
print(f"  Loss: {result['total_loss'].item():.6f}\n")

# Check slot diversity across frames
print("7. Analyzing slot diversity across frames...")
all_slots = result['all_slots'][0]  # (T=8, K=5, D=64)
for frame_idx in range(8):
    slots = all_slots[frame_idx]  # (5, 64)
    slots_norm = torch.nn.functional.normalize(slots, dim=-1)
    sim_matrix = torch.mm(slots_norm, slots_norm.t())
    off_diag = (sim_matrix.sum() - sim_matrix.trace()) / (5 * 4)
    print(f"  Frame {frame_idx}: slot similarity = {off_diag.item():.4f}")

print("\n✅ All tests passed!")
print("\nNext steps:")
print("  1. Run: bash src/train_video_ablations.sh")
print("  2. Compare: logs/video_*.log")
print("  3. Visualize: python src/compare_video_ablations.py")
