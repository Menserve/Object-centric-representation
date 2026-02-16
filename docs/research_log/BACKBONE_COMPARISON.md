# ViT Backbone Comparison Report: Feature Normalization Fix

## Executive Summary

Successfully trained 3 ViT backbones (DINOv2, DINOv1, CLIP) for object-centric learning on specular/metal objects. **Critical discovery**: Feature normalization is essential to handle scale differences across backbones.

## Problem Diagnosis

Initial training failed for DINOv1 and CLIP despite using correct batch_size=2:

| Backbone | Feature Std | Initial Training | Result |
|----------|-------------|------------------|--------|
| DINOv2 | 2.41 | Loss: 0.732 | ✅ Success |
| DINOv1 | 3.74 (55% larger) | Loss: 3.903 | ❌ Failed |
| CLIP | 0.47 (80% smaller) | Loss: -0.014 | ❌ Failed |

**Root Cause**: Large variance in feature distributions across backbones destabilized Slot Attention training.

## Solution: Feature Normalization

Added per-sample feature normalization in `FeatureExtractor.forward()`:

```python
# Normalize to zero mean and unit std
mean = features.mean(dim=(2, 3), keepdim=True)
std = features.std(dim=(2, 3), keepdim=True) + 1e-6
features = (features - mean) / std
```

This ensures all backbones produce features with consistent scale before Slot Attention processing.

## Training Results (with Feature Normalization)

### Training Loss Comparison

| Backbone | Architecture | Best Loss | Test Loss | Training Speed |
|----------|-------------|-----------|-----------|----------------|
| **DINOv2** | ViT-S/14 (384d) | 0.261609 | 0.620131 | 0.7s/epoch |
| **DINOv1** | ViT-S/16 (384d) | 0.260078 | 0.746576 | 0.9s/epoch |
| **CLIP** | ViT-B/16 (768d→384d) | 0.021885 | 0.036011 | 0.4s/epoch |

### Key Observations

1. **DINOv2 vs DINOv1**: Nearly identical training loss (~0.26), but DINOv1 has higher test loss (0.75 vs 0.62), suggesting slight overfitting or different feature generalization.

2. **CLIP**: Significantly lower loss (0.022) compared to DINO variants (0.26). This could indicate:
   - Better feature representation for this task
   - Different loss landscape (needs visual validation)
   - Potential over-smoothing (check slot diversity)

3. **Training Speed**: CLIP trains faster (0.4s/epoch) despite larger backbone (768d→384d), likely due to more efficient forward pass.

## Configuration Details

### Common Settings
- **Dataset**: MOVi-A subset (60 samples: 20 metal + 40 mixed)
- **Batch Size**: 2 (critical for stability)
- **Learning Rate**: 0.0004 with 5-epoch warmup + cosine decay
- **Epochs**: 200
- **Temperature Scaling**: τ=0.5 (23% improvement over τ=1.0)
- **Diversity Weight**: 0.1
- **Data Loading**: num_workers=8, pin_memory=True
- **GPU**: RTX 5090 (~60% utilization)

### Backbone-Specific Details

**DINOv2 (dinov2_vits14)**:
- Feature extractor: `forward_features()` → x_norm_patchtokens
- Spatial resolution: 16×16 (native)
- No projection needed

**DINOv1 (dino_vits16)**:
- Feature extractor: `get_intermediate_layers()` → last layer features
- Spatial resolution: 14×14 → upsampled to 16×16
- No projection needed

**CLIP (clip_vitb16)**:
- Feature extractor: Custom patch-level extraction from visual transformer
- Projection: 768d → 384d (Conv2d 1×1)
- Spatial resolution: 14×14 → upsampled to 16×16

## Lessons Learned

### What Worked ✅
1. **Feature Normalization**: Absolutely critical for multi-backbone comparison
2. **Temperature Scaling (τ=0.5)**: 23% improvement in mask differentiation
3. **2-Layer MLP Projection**: Prevents slot collapse
4. **Xavier Initialization**: Provides proper gradient flow
5. **Data Loading Optimization**: num_workers=8 improved GPU usage from 14% to 60%

### What Failed ❌
1. **Large Batch Sizes (16/12)**: Caused training instability despite faster execution
2. **Video Mode (SAVi)**: Slot Predictor collapse (future work)
3. **No Feature Normalization**: 5× loss degradation for DINOv1, negative loss for CLIP

### Critical Insight
**Hardware optimization (GPU usage, training speed) must not compromise output quality.** 

Initial attempt to maximize GPU utilization by increasing batch size achieved:
- GPU: 14% → 87% ✅
- Speed: 2.7s → 0.3s/epoch ✅
- **Quality: Loss 0.7 → 4.0** ❌

This failure taught us to:
1. Always validate output quality, not just hardware metrics
2. Be suspicious of unusually fast convergence
3. Prioritize stability over speed for small datasets

## Next Steps

### Immediate (待機中)
1. ✅ Visual inspection of slot masks (user checking results)
2. ⏳ Quantitative comparison of slot diversity metrics
3. ⏳ Generate final comparison figures

### Future Work
- [ ] Test on full MOVi-A dataset (verify scaling behavior)
- [ ] Investigate why CLIP has lower loss (validate slot quality)
- [ ] Explore video mode with feature normalization
- [ ] Fine-tune backbone-specific learning rates
- [ ] Document in TA report with failure analysis

## Checkpoints

- `checkpoints/dinov2_normalized/dinov2_vits14/best_model.pt`
- `checkpoints/dinov1_normalized/dino_vits16/best_model.pt`
- `checkpoints/clip_normalized/clip_vitb16/best_model.pt`

## Reproducibility

All training logs saved in:
- `logs/dinov2_normalized.log`
- `logs/dinov1_normalized.log`
- `logs/clip_normalized.log`

Command template:
```bash
python src/train_movi.py \
  --backbone {dinov2_vits14|dino_vits16|clip_vitb16} \
  --data_dir data/movi_a_subset \
  --num_epochs 200 \
  --max_frames 1 \
  --mask_temperature 0.5 \
  --batch_size 2 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --save_dir checkpoints/{name}_normalized
```

**Note**: Feature normalization is now built into `src/savi_dinosaur.py` (FeatureExtractor.forward, lines 143-148).

---

*Generated: 2026-02-15 21:32 (JST)*  
*Feature normalization fix dramatically improved training stability across all backbones.*
