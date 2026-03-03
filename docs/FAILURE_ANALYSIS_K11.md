# DINOv1/CLIP K=11 Failure Analysis

**Date**: 2026-03-02  
**Author**: SAVi-DINOSAUR Project  
**Context**: At K=11, DINOv2 jumps from FG-ARI 0.155→0.470, but DINOv1 drops (0.176→0.131) and CLIP barely improves (0.047→0.110). Why?

---

## Executive Summary

Three distinct failure modes explain the backbone divergence:

| Backbone | Failure Mode | Root Cause | Evidence |
|---|---|---|---|
| **DINOv2** | ✅ Success | "Goldilocks" spatial features | FG-ARI=0.470, 10.3/11 slots active |
| **DINOv1** | Slot collapse (1 slot dominates) | channel_norm loss + high feature complexity | Recon MSE ≥ mean-fill baseline |
| **CLIP** | Uniform masks (no specialization) | Spatially uniform features | Converged at epoch 4, recon 74% worse than mean-fill |

---

## 1. Feature Space Properties (Frozen Backbone Output)

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Feature dim | 384 | 384 | 768→384 (projected) |
| Patch size | 14 | 16 | 16 |
| Spatial resolution | 16×16 | 14×14 | 14×14 |
| Mean feature norm | 45-46 | 80-84 | 8.7-8.9 |
| Spatial variance | 2.67 | **11.59** (highest) | 0.039 (lowest) |
| Cosine similarity (raw) | 0.52 | **0.34** (most diverse) | **0.87** (most uniform) |
| PCA dims for 90% variance | 14.6 | 25.2 | 2.3 |

**Key insight**: DINOv1 has the MOST spatially diverse features, yet fails. CLIP has the LEAST diverse features, and also fails. DINOv2 sits in a "sweet spot."

## 2. Projection Bottleneck (384→64)

The `feature_projection` MLP (LayerNorm → Linear(384,384) → ReLU → Linear(384,64)) transforms features before Slot Attention:

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Raw cosine sim | 0.51 | 0.34 | 0.92 |
| **Projected cosine sim** | **0.29** ↓ better | **0.17** ↓ better | **0.98** ↑ **worse** |
| Projected variance | 0.217 | 0.542 | 0.013 |
| Projected norm | 4.18 | 5.64 | 7.75 |

**Critical**: For CLIP, the learned projection makes features EVEN MORE uniform (0.92→0.98). The LayerNorm weights are barely trained (std=0.002 vs DINOv2's 0.034). The mapping simply amplifies the uniformity — Slot Attention receives essentially identical vectors at every position.

For DINOv1/DINOv2, projection actually IMPROVES spatial diversity (cosine similarity decreases).

## 3. Slot Representation Analysis

After Slot Attention (5 iterations), slot vectors show:

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Slot pairwise cosine | **0.71** (similar) | **-0.005** (diverse) | 0.056 |
| Slot vector std | 0.32 | **2.61** (highest) | 0.58 |

**Counterintuitive**: DINOv2 slots are MOST similar to each other (cos=0.71), yet produce the BEST segmentation. DINOv1 slots are maximally diverse (cos≈0), yet collapse to one dominant slot.

**Explanation**: Slot diversity in representation space ≠ spatial specialization. DINOv2 slots are similar because they all live in a compact subspace but differ in their attention patterns (which positions they attend to). DINOv1 slots are diverse but only one slot successfully captures the attention signal — the others are pushed to irrelevant subspaces.

## 4. Mask Quality (50-scene aggregate)

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Active slots (>1%) | **10.3/11** | 5.4/11 | 3.6/11 |
| Top slot area | 44.9% | **72.8%** (dominant) | **70.2%** (dominant) |
| Mask entropy | 1.465 | 1.072 (collapsed) | **2.121** (uniform) |
| Entropy / max | 0.61 | 0.45 | **0.88** (near-max) |
| Gini coefficient | 0.585 | 0.817 | 0.827 |

DINOv1 and CLIP both show extreme inequality (Gini>0.8) but for opposite reasons:
- **DINOv1**: Low entropy → one slot is "sharp" and covers most of the image
- **CLIP**: High entropy → all masks are near-uniform (each slot covers ~1/11 everywhere)

## 5. Reconstruction Quality

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Reconstruction MSE | 1.076 | 14.566 | 0.076 |
| Mean-fill MSE (trivial) | 2.699 | 14.152 | 0.044 |
| Ratio (recon/trivial) | **0.40** ✅ | **1.03** ❌ | **1.74** ❌ |

- **DINOv2**: 60% better than trivial → real spatial learning
- **DINOv1**: Equal to trivial → no spatial information captured despite 199 epochs
- **CLIP**: 74% WORSE than trivial → model adds noise, found degenerate solution

## 6. Training Dynamics

| Metric | DINOv2 | DINOv1 | CLIP |
|---|---|---|---|
| Best epoch | 195 | 199 | **4** |
| Loss type | MSE | channel_norm | MSE |
| Best loss | 0.907 | 0.269 | 0.057 |

**CLIP converged at epoch 4**: Because features are near-uniform (spatial variance=0.039), the trivial "predict mean" solution already has MSE≈0.04. The model can't find a better solution with 11 slots, so it stops improving immediately.

## 7. Root Cause Synthesis

### CLIP: "Nothing to Decompose"

```
CLIP features → spatially uniform (cos_sim=0.87)
  → Projection amplifies uniformity (cos_sim→0.98)  
  → Slot Attention receives identical inputs at every position
  → Attention weights become uniform (no position can be distinguished)
  → All 11 masks ≈ 1/11 everywhere (entropy near maximum)
  → Reconstruction = mean feature ≈ trivial solution
  → Loss converges at epoch 4 with degenerate solution
```

**Why**: CLIP is trained for image-level semantic alignment (text-image matching), NOT spatial discrimination. Its features encode "what category" not "where objects are." This is fundamental — no amount of Slot Attention iterations can create spatial structure from spatially uniform input.

### DINOv1: "Too Complex to Project" 

```
DINOv1 features → highly spatially diverse (cos_sim=0.34, var=11.59)
  BUT → high intrinsic dimensionality (25 PCA dims for 90%)
  AND → 2× feature magnitude vs DINOv2 (norm=80 vs 45)
  → 384→64 projection loses critical spatial modes
  → channel_norm loss optimizes in normalized space
  → Model achieves low channel_norm loss (0.269) but raw MSE ≥ trivial
  → One "background" slot dominates (72.8% area, 5-7 active slots)
  → Diversity regularization can't overcome the optimization landscape
```

**Why**: DINOv1 features are spatially rich but too high-dimensional for the 64-dim slot space. The 384→64 bottleneck discards ~60% of the variance (DINOv1 needs 25 PCA dims vs DINOv2's 15). Combined with channel_norm loss (which can be minimized without faithful spatial reconstruction), the model finds a degenerate single-slot solution where one slot reconstructs a blurred average.

### DINOv2: "Goldilocks" Properties

```
DINOv2 features → moderate spatial diversity (cos_sim=0.52, var=2.67)
  → compact structure (15 PCA dims for 90%)
  → projection IMPROVES diversity (cos_sim: 0.52→0.29)
  → reasonable scale (norm=45)
  → Slot Attention can distinguish positions via attention
  → 10.3/11 slots become active with distinct spatial regions
  → Reconstruction MSE 60% better than trivial
  → FG-ARI 0.470
```

DINOv2's self-supervised objective (iBOT + DINOv2 loss) creates features that are:
1. **Spatially discriminative** enough for Slot Attention to differentiate positions
2. **Compact** enough to survive 384→64 projection without losing critical information
3. **Appropriately scaled** for the downstream optimization landscape

---

## 8. Potential Fixes

### For CLIP (hard to fix):
- The spatial uniformity is fundamental to CLIP's architecture
- Would need CLIP features from intermediate layers (not final) or spatial CLIP variants
- **Verdict**: Likely unfixable without changing the backbone or using spatial feature maps

### For DINOv1 (potentially fixable):
1. **Increase slot_dim**: 64→128 or 192 to accommodate higher-dimensional features
2. **Use MSE loss instead of channel_norm**: Channel_norm may mask the spatial failure
3. **Feature normalization before projection**: Scale DINOv1 features to match DINOv2's range
4. **PCA pre-projection**: Reduce 384 to ~25 principal components before the MLP
5. **Train longer with lower LR**: May need different optimization dynamics

### Priority for poster:
These fixes are outside scope for the 3/8 poster. The finding itself — that **backbone feature spatial structure determines Slot Attention success** — is the key contribution.
