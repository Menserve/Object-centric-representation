# Architecture Fix Report
**Date**: 2025-02-01  
**Problem**: Slot Attention produces flat colored patches instead of object segmentation  
**Root Cause**: 3 critical architecture gaps compared to OCL Framework reference implementation

---

## Background

After 7+ debugging iterations (Xavier init fix improved Over-smoothing metrics but visual output unchanged), deep analysis of OCL Framework's DINOSAUR implementation revealed we were missing 3 key architectural components.

**Key Insight**: Gemini's suggestion to add Positional Embedding was INCORRECT for DINOSAUR. The official DINOSAUR config uses `DummyPositionEmbed` (pass-through) because ViT features already contain positional information from the ViT's own positional encoding. CNN-based Slot Attention needs `SoftPositionEmbed`, but ViT-based DINOSAUR does not.

---

## Critical Differences Found

### 1. Feature Projection MLP (MOST CRITICAL)

| Component | Our Implementation | DINOSAUR Reference |
|---|---|---|
| **Structure** | Single Linear Layer | Two-Layer MLP |
| **Formula** | `LayerNorm(384) → Linear(384→64)` | `LayerNorm(384) → Linear(384→384) → ReLU → Linear(384→64)` |
| **Non-linearity** | None | ReLU |
| **Expressiveness** | Linear projection only | Can learn complex non-linear feature transformations |

**Why This Matters**: DINOv2 features (B, 256, 384) have rich spatial structure. A single linear projection to 64-dim can destroy spatial variance. The two-layer MLP with ReLU preserves spatial differentiation through non-linear mapping.

**Evidence from OCL Framework**:
```yaml
# configs/experiment/projects/bridging/dinosaur/_base_feature_recon.yaml
positional_embedding:
  _target_: ocl.neural_networks.wrappers.Sequential
  _args_:
    - _target_: ocl.neural_networks.positional_embedding.DummyPositionEmbed  # Pass-through
    - _target_: ocl.neural_networks.build_two_layer_mlp
      input_dim: ${experiment.input_feature_dim}  # 384
      output_dim: ${....feature_dim}  # e.g., 128 or 64
      hidden_dim: ${experiment.input_feature_dim}  # 384
      initial_layer_norm: true
```

### 2. ff_mlp Hidden Dimension (MEDIUM)

| Component | Our Implementation | OCL Reference |
|---|---|---|
| **Hidden dim** | 128 (2×slot_dim) | 4×slot_dim |
| **For slot_dim=64** | 128 | 256 |

**Why This Matters**: The feed-forward MLP in Slot Attention refines slot representations after GRU updates. A larger hidden dimension (4×) provides more representational capacity.

**Evidence from OCL Framework**:
```yaml
# configs/experiment/slot_attention/_base.yaml
ff_mlp:
  _target_: ocl.neural_networks.build_two_layer_mlp
  input_dim: 64
  output_dim: 64
  hidden_dim: 128  # For object_dim=64
  initial_layer_norm: true
  residual: true

# For DINOSAUR with object_dim=128:
# hidden_dim: "${eval_lambda:'lambda dim: 4 * dim', ${..object_dim}}"  # = 512
```

### 3. kvq_dim for Attention Space (MEDIUM)

| Component | Our Implementation | OCL Base Config |
|---|---|---|
| **Keys/Values/Queries dim** | Same as slot_dim (64) | 2× slot_dim (128) |
| **Attention space** | 64-dimensional | 128-dimensional |

**Why This Matters**: Larger kvq_dim gives the attention mechanism more expressive power to distinguish between features.

**Evidence from OCL Framework**:
```yaml
# configs/experiment/slot_attention/_base.yaml
perceptual_grouping:
  feature_dim: 64
  object_dim: 64
  kvq_dim: 128  # 2× larger!
```

---

## Implementation

### Modified Files
- `src/savi_dinosaur.py` (3 replacements)

### Changes Made

#### 1. SlotAttentionSAVi.__init__ - Added feature_dim and kvq_dim parameters

```python
def __init__(
    self,
    num_slots: int,
    dim: int,
    feature_dim: Optional[int] = None,  # NEW: Allows different input feature dim
    kvq_dim: Optional[int] = None,      # NEW: Separate attention space dim
    iters: int = 5,
    hidden_dim: int = 512,
    eps: float = 1e-8
):
    # ...
    self.feature_dim = feature_dim if feature_dim is not None else dim
    self.kvq_dim = kvq_dim if kvq_dim is not None else dim
    self.scale = (self.kvq_dim // 1) ** -0.5
    
    # K, V project from feature_dim; Q projects from dim
    self.to_q = nn.Linear(dim, self.kvq_dim, bias=False)
    self.to_k = nn.Linear(self.feature_dim, self.kvq_dim, bias=False)
    self.to_v = nn.Linear(self.feature_dim, self.kvq_dim, bias=False)
    
    self.gru = nn.GRUCell(self.kvq_dim, dim)  # kvq_dim → dim
```

#### 2. SlotAttentionSAVi.forward - Updated for kvq_dim

```python
k = self.to_k(inputs)  # (B, N, kvq_dim)
v = self.to_v(inputs)  # (B, N, kvq_dim)

# ...
q = self.to_q(slots)  # (B, K, kvq_dim)

# GRU update: kvq_dim → dim
slots = self.gru(
    updates.reshape(-1, self.kvq_dim),
    slots_prev.reshape(-1, self.dim)
)
```

#### 3. SAViDinosaur.__init__ - Two-layer MLP + larger hidden dims

```python
# ★ Fix 1: Two-layer MLP for feature projection
self.feature_projection = nn.Sequential(
    nn.LayerNorm(feat_dim),
    nn.Linear(feat_dim, feat_dim),         # 384 → 384
    nn.ReLU(inplace=True),                 # Non-linearity!
    nn.Linear(feat_dim, slot_dim)          # 384 → 64
)

# ★ Fix 2 & 3: kvq_dim=128, hidden_dim=256 (4×slot_dim)
self.slot_attention = SlotAttentionSAVi(
    num_slots, 
    dim=slot_dim,              # 64
    feature_dim=slot_dim,      # 64 (after projection)
    kvq_dim=128,               # 2× larger attention space
    iters=5, 
    hidden_dim=256             # 4× for ff_mlp
)
self.slot_predictor = SlotPredictor(slot_dim, hidden_dim=256)
```

---

## Expected Impact

### Before (Xavier init only):
- **Iteration dynamics**: Similarity 0.57-0.58 (still too high)
- **Entropy**: Increases 40→60 (should decrease!)
- **Visual output**: All 5 slots are flat colored patches
- **Loss**: Converges to ~1.37

### After (Xavier init + 3 architecture fixes):
- **Feature projection**: Non-linear MLP preserves spatial variance in projected features
- **Attention mechanism**: Larger kvq_dim (128 vs 64) → richer attention space
- **Slot representation**: Larger ff_mlp hidden_dim (256 vs 128) → better slot refinement
- **Expected**: Attention maps should differentiate objects, slots should produce object contours

---

## Next Steps

1. **Retrain**: 50 epochs, single-frame mode (max_frames=1), to isolate spatial attention
2. **Verify with debug tools**:
   - `src/analyze_iteration_dynamics.py`: Check similarity <0.3, entropy decreases
   - Visual inspection: Do masks show object contours?
3. **If successful**: Re-enable temporal (max_frames=12), then multi-backbone comparison
4. **If still fails**: Consider increasing slot_dim from 64→128 (DINOSAUR MOVi-C uses 128)

---

## References

- **OCL Framework DINOSAUR Base Config**: `ocl_framework/configs/experiment/projects/bridging/dinosaur/_base_feature_recon.yaml`
- **OCL Framework Slot Attention Base**: `ocl_framework/configs/experiment/slot_attention/_base.yaml`
- **DINOSAUR Paper**: Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning", ICLR 2023
- **OCL Framework GitHub**: https://github.com/amazon-science/object-centric-learning-framework
