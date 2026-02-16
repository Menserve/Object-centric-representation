"""Ground truth verification for old checkpoint format (single Linear slot_to_feature)"""
import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
import numpy as np

# Load checkpoint
checkpoint_path = '../checkpoints/twolayer_mlp_200ep/dinov2_vits14/best_model.pt'
print(f"Loading checkpoint: {checkpoint_path}")

# Create model with current architecture
model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)

# Load checkpoint with strict=False to check what's missing
checkpoint = torch.load(checkpoint_path, map_location='cpu')
incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(f"\nMissing keys: {len(incompatible.missing_keys)}")
print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

# Check if the incompatibility is only in slot_to_feature
if incompatible.unexpected_keys:
    print("\nUnexpected keys (old architecture):")
    for key in incompatible.unexpected_keys:
        if 'slot_to_feature' in key:
            print(f"  {key}: shape {checkpoint['model_state_dict'][key].shape}")

if incompatible.missing_keys:
    print("\nMissing keys (new architecture):")
    for key in incompatible.missing_keys:
        if 'slot_to_feature' in key:
            print(f"  {key}")

# Manually copy slot_to_feature weights
print("\n" + "="*60)
print("WORKAROUND: Converting old Linear to new 2-layer MLP")
print("="*60)

old_weight = checkpoint['model_state_dict']['slot_to_feature.weight']  # (384, 64)
old_bias = checkpoint['model_state_dict']['slot_to_feature.bias']      # (384,)

print(f"Old Linear: weight shape {old_weight.shape}, bias shape {old_bias.shape}")

# Strategy: Initialize the 2-layer MLP and manually set the second layer
# to mimic the old single linear as closely as possible
# Layer 0: LayerNorm (keep random init)
# Layer 1: Linear(64, 128) → keep identity-like
# Layer 2: ReLU
# Layer 3: Linear(128, 384) → use old weights with appropriate reshaping

with torch.no_grad():
    # Keep LayerNorm as trained
    # Set first linear to identity-like (this is a new layer that didn't exist)
    model.slot_to_feature[1].weight[:64, :64] = torch.eye(64)
    model.slot_to_feature[1].weight[64:, :64] = torch.zeros(64, 64)
    model.slot_to_feature[1].bias.fill_(0.0)
    
    # Set second linear to match old weights (approximately)
    # old: y = Wx + b where W is (384, 64)
    # new: z = ReLU(Ax), y = Bz where A is (128, 64) and B is (384, 128)
    # Approximate: B @ A ≈ W, so we can use B = W[:, :128] and adjust A
    
    # Simple approach: Use the old weights in the second layer
    # B should be (384, 128), but old_weight is (384, 64)
    # Pad with zeros or duplicate
    model.slot_to_feature[3].weight[:, :64] = old_weight
    model.slot_to_feature[3].weight[:, 64:] = torch.zeros(384, 64)
    model.slot_to_feature[3].bias = old_bias

print("\nApproximate conversion complete. This won't be identical but gives a rough idea.")
print("NOTE: Better approach is to retrain with fixed architecture!\n")

# Load test data
print("Loading test data...")
test_data = torch.load('../data/movi_a_subset/metal_000.pt')
img = test_data['video'][0, 0:1]  # First frame, shape (1, 3, 256, 256)

# Resize to 224x224
import torch.nn.functional as F
img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

print(f"Test image shape: {img.shape}")

# Forward pass
model.eval()
with torch.no_grad():
    recon, target, masks, slots = model.forward_image(img)

print(f"\nForward pass successful!")
print(f"  - Masks: {masks.shape}")
print(f"  - Slots: {slots.shape}")

# Analyze slot diversity
slots_2d = slots.squeeze(0)  # (K, D)
slot_sim = torch.corrcoef(slots_2d).numpy()
print(f"\n" + "="*60)
print("SLOT VECTOR SIMILARITY (correlation matrix):")
print("="*60)
print(slot_sim)
avg_sim = (slot_sim.sum() - np.trace(slot_sim)) / (slot_sim.shape[0] * (slot_sim.shape[0] - 1))
print(f"\nAverage pairwise slot similarity: {avg_sim:.3f}")

# Analyze mask diversity
masks_flat = masks.squeeze(0).view(masks.shape[1], -1)  # (K, H*W)
mask_corr = torch.corrcoef(masks_flat).numpy()
print(f"\n" + "="*60)
print("MASK SIMILARITY (correlation matrix):")
print("="*60)
print(mask_corr)
avg_mask_sim = (mask_corr.sum() - np.trace(mask_corr)) / (mask_corr.shape[0] * (mask_corr.shape[0] - 1))
print(f"\nAverage pairwise mask similarity: {avg_mask_sim:.3f}")

# Check mask coverage
masks_soft = masks.squeeze().detach()
print(f"\n" + "="*60)
print("MASK COVERAGE (% of image with >0.5 attention):")
print("="*60)
for i in range(masks_soft.shape[0]):
    coverage = (masks_soft[i] > 0.5).float().mean().item() * 100
    print(f"  Slot {i}: {coverage:.1f}%")

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
print(f"Slot vector similarity: {avg_sim:.3f}")
print(f"Mask similarity: {avg_mask_sim:.3f}")
print(f"\n⚠️  NOTE: This used an approximate conversion.")
print(f"    For accurate results, retrain with the fixed 2-layer MLP architecture!")
