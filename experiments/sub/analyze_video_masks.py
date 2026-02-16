"""
Video Mode マスク品質分析
========================
1. スロットの物体担当分析（slot-object assignment）
2. マスクにじみ（bleeding）の定量化
3. 解像度vs計算コスト推定
4. フレーム数による改善予測
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64, mask_temperature=0.5)
ckpt = torch.load('../checkpoints/video_mode_8frames/dinov2_vits14/best_model.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

# Load dataset
dataset = MoviDataset('../data/movi_a_subset', split='metal', max_frames=8)
print(f"Metal samples: {len(dataset)}")

# ==========================================
# 1. Slot-Object Assignment Analysis
# ==========================================
print("\n" + "="*60)
print("1. SLOT-OBJECT ASSIGNMENT ANALYSIS")
print("="*60)

# Analyze 5 metal samples
for sample_idx in range(min(5, len(dataset))):
    sample = dataset[sample_idx]
    video = sample['video'].unsqueeze(0).to(device)  # (1, T, 3, H, W)
    seg_gt = sample['segmentation']  # (T, H, W)
    
    with torch.no_grad():
        result = model.forward_video(video, return_all_masks=True)
    
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    t, k, _, h, w = all_masks.shape
    
    # Resize masks to GT resolution
    masks_resized = F.interpolate(
        all_masks.view(t * k, 1, h, w),
        size=seg_gt.shape[-2:],
        mode='bilinear'
    ).view(t, k, seg_gt.shape[-2], seg_gt.shape[-1])  # (T, K, H, W)
    
    print(f"\n--- Sample {sample_idx}: {sample['filename']} ---")
    
    # Per-frame slot-object IoU
    num_objects = seg_gt.max().item() + 1  # includes background (0)
    
    for frame_idx in [0, t//2, t-1]:
        seg_frame = seg_gt[frame_idx]  # (H, W)
        masks_frame = masks_resized[frame_idx]  # (K, H, W)
        
        print(f"\n  Frame {frame_idx}:")
        # Compute IoU between each slot and each GT object
        iou_matrix = torch.zeros(k, num_objects)
        for slot_idx in range(k):
            slot_mask = (masks_frame[slot_idx] > 0.3).float().cpu()
            for obj_idx in range(num_objects):
                obj_mask = (seg_frame == obj_idx).float().cpu()
                intersection = (slot_mask * obj_mask).sum()
                union = ((slot_mask + obj_mask) > 0).float().sum()
                iou_matrix[slot_idx, obj_idx] = intersection / (union + 1e-8)
        
        for slot_idx in range(k):
            best_obj = iou_matrix[slot_idx].argmax().item()
            best_iou = iou_matrix[slot_idx].max().item()
            coverage = (masks_frame[slot_idx] > 0.3).float().mean().item() * 100
            max_val = masks_frame[slot_idx].max().item()
            print(f"    Slot {slot_idx}: best_obj={best_obj} IoU={best_iou:.3f} "
                  f"coverage={coverage:.1f}% max={max_val:.3f}")
    
    # Temporal consistency: does slot assignment change across frames?
    print(f"\n  Temporal consistency (slot→object mapping):")
    for slot_idx in range(k):
        assigned_objects = []
        for frame_idx in range(t):
            seg_frame = seg_gt[frame_idx]
            masks_frame = masks_resized[frame_idx]
            slot_mask = (masks_frame[slot_idx] > 0.3).float().cpu()
            
            best_iou = 0
            best_obj = -1
            for obj_idx in range(num_objects):
                obj_mask = (seg_frame == obj_idx).float().cpu()
                intersection = (slot_mask * obj_mask).sum()
                union = ((slot_mask + obj_mask) > 0).float().sum()
                iou = (intersection / (union + 1e-8)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj_idx
            assigned_objects.append(best_obj)
        
        # Check consistency
        unique_objects = set(assigned_objects)
        consistency = max(assigned_objects.count(obj) for obj in unique_objects) / t * 100
        print(f"    Slot {slot_idx}: assignments={assigned_objects} "
              f"consistency={consistency:.0f}%")


# ==========================================
# 2. MASK BLEEDING (にじみ) QUANTIFICATION
# ==========================================
print("\n" + "="*60)
print("2. MASK BLEEDING (にじみ) ANALYSIS")
print("="*60)

# Bleeding = how much mask extends beyond the object it's assigned to
sample = dataset[0]
video = sample['video'].unsqueeze(0).to(device)
seg_gt = sample['segmentation']

with torch.no_grad():
    result = model.forward_video(video, return_all_masks=True)

all_masks = result['all_masks'][0]
t, k, _, h, w = all_masks.shape

masks_resized = F.interpolate(
    all_masks.view(t * k, 1, h, w),
    size=seg_gt.shape[-2:],
    mode='bilinear'
).view(t, k, seg_gt.shape[-2], seg_gt.shape[-1])

print("\nMask statistics (raw 16×16 resolution):")
for slot_idx in range(k):
    mask_raw = all_masks[0, slot_idx, 0]  # First frame, (16, 16)
    vals = mask_raw.cpu().numpy().flatten()
    # Entropy: how spread out is the mask?
    entropy = -(vals * np.log(vals + 1e-10)).sum()
    # Sharpness: how binary is the mask?
    sharpness = ((vals - 0.5)**2).mean() * 4  # 0 = uniform, 1 = binary
    # Effective resolution
    above_threshold = (vals > 0.1).sum()
    print(f"  Slot {slot_idx}: mean={vals.mean():.4f} std={vals.std():.4f} "
          f"max={vals.max():.4f} entropy={entropy:.2f} "
          f"sharpness={sharpness:.4f} active_pixels={above_threshold}/256")

# Bleeding score per slot
print("\nBleeding scores (fraction of mask weight outside assigned object):")
for frame_idx in [0, 3, 7]:
    seg_frame = seg_gt[frame_idx]
    masks_frame = masks_resized[frame_idx]
    
    print(f"\n  Frame {frame_idx}:")
    for slot_idx in range(k):
        slot_mask = masks_frame[slot_idx].cpu()
        
        # Find assigned object (highest overlap)
        best_obj = -1
        best_overlap = 0
        num_objects = seg_frame.max().item() + 1
        for obj_idx in range(num_objects):
            obj_mask = (seg_frame == obj_idx).float().cpu()
            overlap = (slot_mask * obj_mask).sum().item()
            if overlap > best_overlap:
                best_overlap = overlap
                best_obj = obj_idx
        
        # Bleeding = mask weight outside assigned object
        obj_mask = (seg_frame == best_obj).float().cpu()
        total_weight = slot_mask.sum().item()
        inside_weight = (slot_mask * obj_mask).sum().item()
        bleeding = 1 - inside_weight / (total_weight + 1e-8)
        
        print(f"    Slot {slot_idx} → obj {best_obj}: "
              f"bleeding={bleeding:.1%} "
              f"(inside={inside_weight:.1f}, total={total_weight:.1f})")


# ==========================================
# 3. RESOLUTION & COST ANALYSIS
# ==========================================
print("\n" + "="*60)
print("3. RESOLUTION vs COMPUTATIONAL COST")
print("="*60)

print("""
Current architecture (16×16 mask resolution):
  - ViT-S/14: 224/14 = 16 patches → 16×16 feature map
  - Spatial Broadcast Decoder: 16×16 output
  - Total mask pixels per slot: 256

にじみの主な原因:
  [1] 低解像度 (16×16): 1 patch = 14×14 pixels → 1ピクセルが196pixel²をカバー
  [2] Softmax競合: 全スロットが全位置で競合 → 境界が曖昧
  [3] Bilinear upsampling: 16×16→224×224 で境界がぼける

解像度を上げる方法と計算コスト:
""")

# Compute actual cost estimates
def estimate_cost(h, w, feat_dim=384, num_slots=5, decoder_channels=384):
    """Estimate FLOPs for different decoder resolutions"""
    # Spatial broadcast: (K, D+2, H, W)
    spatial_broadcast = num_slots * (feat_dim + 2) * h * w
    
    # 3-layer CNN decoder
    # Conv1: (D+2) → 384, 5×5
    conv1_flops = num_slots * h * w * (feat_dim + 2) * decoder_channels * 25
    # Conv2: 384 → 384, 5×5
    conv2_flops = num_slots * h * w * decoder_channels * decoder_channels * 25
    # Conv3: 384 → (D+1), 3×3
    conv3_flops = num_slots * h * w * decoder_channels * (feat_dim + 1) * 9
    
    total_flops = conv1_flops + conv2_flops + conv3_flops
    return total_flops

# Current: 16×16
cost_16 = estimate_cost(16, 16)
# Hypothetical: 32×32
cost_32 = estimate_cost(32, 32)
# Hypothetical: 64×64
cost_64 = estimate_cost(64, 64)
# Hypothetical: 224×224 (pixel-level)
cost_224 = estimate_cost(224, 224)

print(f"  16×16 (current):     {cost_16/1e9:.2f} GFLOPs  (1.0×)")
print(f"  32×32 (4× pixels):   {cost_32/1e9:.2f} GFLOPs  ({cost_32/cost_16:.1f}×)")
print(f"  64×64 (16× pixels):  {cost_64/1e9:.2f} GFLOPs  ({cost_64/cost_16:.1f}×)")
print(f"  224×224 (pixel-lvl): {cost_224/1e9:.2f} GFLOPs  ({cost_224/cost_16:.1f}×)")

# Memory estimate
def estimate_memory_mb(h, w, feat_dim=384, num_slots=5, batch_size=2, num_frames=8):
    # Decoder intermediate activations per frame
    # Each conv layer: (B*K, C, H, W) × 4 bytes
    per_frame = batch_size * num_slots * feat_dim * h * w * 4 * 3  # 3 conv layers
    total = per_frame * num_frames
    return total / (1024**2)

mem_16 = estimate_memory_mb(16, 16)
mem_32 = estimate_memory_mb(32, 32)
mem_64 = estimate_memory_mb(64, 64)

print(f"\n  Memory (activations, B=2, T=8):")
print(f"  16×16: {mem_16:.0f} MB")
print(f"  32×32: {mem_32:.0f} MB")
print(f"  64×64: {mem_64:.0f} MB")


# ==========================================
# 4. FRAME COUNT vs ASSIGNMENT QUALITY
# ==========================================
print("\n" + "="*60)
print("4. FRAME COUNT vs MASK QUALITY")
print("="*60)

print("\nAnalyzing how slot assignment evolves across frames...")

# Use first metal sample with all 24 frames
dataset_full = MoviDataset('../data/movi_a_subset', split='metal', max_frames=24)
sample_full = dataset_full[0]
video_full = sample_full['video'].unsqueeze(0).to(device)

with torch.no_grad():
    result_full = model.forward_video(video_full, return_all_masks=True)

all_masks_full = result_full['all_masks'][0]  # (24, 5, 1, 16, 16)
all_slots_full = result_full['all_slots'][0]  # (24, 5, 64)
seg_gt_full = sample_full['segmentation']

t_full = all_masks_full.shape[0]

# Track slot diversity and assignment stability over frames
print("\nSlot diversity (mean pairwise cosine similarity) per frame:")
for frame_idx in range(0, t_full, 4):
    slots_f = all_slots_full[frame_idx]  # (5, 64)
    slots_norm = F.normalize(slots_f, dim=-1)
    sim_matrix = torch.mm(slots_norm, slots_norm.t())
    
    # Off-diagonal mean
    mask = 1 - torch.eye(5, device=device)
    off_diag_mean = (sim_matrix * mask).sum() / (5 * 4)
    
    # Mask sharpness
    masks_f = all_masks_full[frame_idx, :, 0]  # (5, 16, 16)
    sharpness = ((masks_f - 0.2) * (masks_f > 0.2).float()).mean().item()
    max_vals = masks_f.max(dim=-1)[0].max(dim=-1)[0]  # (5,)
    
    print(f"  Frame {frame_idx:2d}: slot_sim={off_diag_mean:.4f} "
          f"max_activations=[{', '.join(f'{v:.3f}' for v in max_vals.cpu().numpy())}]")

# Show how the same slot's IoU changes across frames
print("\nSlot-object IoU evolution (Sample 0):")
masks_full_resized = F.interpolate(
    all_masks_full.view(t_full * 5, 1, 16, 16),
    size=seg_gt_full.shape[-2:],
    mode='bilinear'
).view(t_full, 5, seg_gt_full.shape[-2], seg_gt_full.shape[-1])

num_objects = seg_gt_full.max().item() + 1

for slot_idx in range(5):
    ious_per_frame = []
    assigned_per_frame = []
    for frame_idx in range(t_full):
        seg_frame = seg_gt_full[frame_idx]
        slot_mask = (masks_full_resized[frame_idx, slot_idx] > 0.3).float()
        
        best_iou = 0
        best_obj = -1
        for obj_idx in range(num_objects):
            obj_mask = (seg_frame == obj_idx).float().cpu()
            intersection = (slot_mask.cpu() * obj_mask).sum().item()
            union = ((slot_mask.cpu() + obj_mask) > 0).float().sum().item()
            iou = intersection / (union + 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_obj = obj_idx
        ious_per_frame.append(best_iou)
        assigned_per_frame.append(best_obj)
    
    # Summary
    unique = set(assigned_per_frame)
    dominant = max(unique, key=lambda x: assigned_per_frame.count(x))
    consistency = assigned_per_frame.count(dominant) / t_full * 100
    avg_iou = np.mean(ious_per_frame)
    print(f"  Slot {slot_idx}: dominant_obj={dominant} consistency={consistency:.0f}% "
          f"avg_IoU={avg_iou:.3f} "
          f"IoU_trend=[{ious_per_frame[0]:.3f}→{ious_per_frame[t_full//2]:.3f}→{ious_per_frame[-1]:.3f}]")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
結論:
1. スロット担当分離: 上記のslot→object assignmentを参照
2. にじみの原因: 16×16解像度の限界 + bilinear upsampling
3. 計算コスト: 32×32で4×、64×64で16× (現実的なのは32×32まで)
4. フレーム数: 時間的手がかりが担当を安定化させるが、にじみは解消しない
""")
