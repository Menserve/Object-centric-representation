"""
特徴量の飽和チェック：DINOv2 → 2層MLP後の特徴量の標準偏差を確認
"""
import torch
import sys
sys.path.insert(0, '.')
from savi_dinosaur import SAViDinosaur
from train_movi import MoviDataset
import numpy as np

# モデル読み込み
checkpoint_path = '../checkpoints/twolayer_mlp_200ep/dinov2_vits14/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# データ読み込み
dataset = MoviDataset('../data/movi_a_subset', split='all', max_frames=1)
sample = dataset[0]
video = sample['video'].unsqueeze(0)  # (1, T, 3, H, W)

with torch.no_grad():
    # DINOv2特徴量
    dino_features = model.feature_extractor(video[:, 0])  # (1, 384, 16, 16)
    print("🔍 DINOv2 features:")
    print(f"  Shape: {dino_features.shape}")
    print(f"  Mean: {dino_features.mean().item():.6f}")
    print(f"  Std: {dino_features.std().item():.6f}")
    print(f"  Min: {dino_features.min().item():.6f}")
    print(f"  Max: {dino_features.max().item():.6f}")
    
    # Reshape to (B, N, D)
    b, c, h, w = dino_features.shape
    features_perm = dino_features.permute(0, 2, 3, 1)  # (1, 16, 16, 384)
    features_flat = features_perm.reshape(b, -1, c)  # (1, 256, 384)
    
    print("\n🔍 Flattened features (before projection):")
    print(f"  Shape: {features_flat.shape}")
    print(f"  Mean: {features_flat.mean().item():.6f}")
    print(f"  Std: {features_flat.std().item():.6f}")
    
    # 2層MLP投影後
    features_projected = model.feature_projection(features_flat)  # (1, 256, 64)
    
    print("\n🔍 Projected features (after 2-layer MLP):")
    print(f"  Shape: {features_projected.shape}")
    print(f"  Mean: {features_projected.mean().item():.6f}")
    print(f"  Std: {features_projected.std().item():.6f}")
    print(f"  Min: {features_projected.min().item():.6f}")
    print(f"  Max: {features_projected.max().item():.6f}")
    
    # 各パッチベクトルの標準偏差（パッチ間の多様性）
    patch_stds = features_projected[0].std(dim=1)  # (256,) - 各パッチの64次元特徴の標準偏差
    print(f"\n📊 Per-patch std (diversity within each patch):")
    print(f"  Mean: {patch_stds.mean().item():.6f}")
    print(f"  Std: {patch_stds.std().item():.6f}")
    print(f"  Min: {patch_stds.min().item():.6f}")
    print(f"  Max: {patch_stds.max().item():.6f}")
    
    # パッチ間のコサイン類似度（全パッチが同じベクトルになっていないか）
    features_norm = torch.nn.functional.normalize(features_projected[0], dim=1)  # (256, 64)
    similarity_matrix = torch.mm(features_norm, features_norm.t())  # (256, 256)
    
    # 対角以外の類似度
    mask = torch.eye(256) == 0
    off_diag_similarities = similarity_matrix[mask]
    
    print(f"\n📊 Patch-to-patch cosine similarity (off-diagonal):")
    print(f"  Mean: {off_diag_similarities.mean().item():.6f}")
    print(f"  Std: {off_diag_similarities.std().item():.6f}")
    print(f"  Min: {off_diag_similarities.min().item():.6f}")
    print(f"  Max: {off_diag_similarities.max().item():.6f}")
    
    if off_diag_similarities.mean().item() > 0.95:
        print("\n❌ PROBLEM: All patches are nearly identical (similarity > 0.95)")
        print("   The 2-layer MLP is collapsing features into a single point!")
    elif off_diag_similarities.mean().item() > 0.8:
        print("\n⚠️  WARNING: Patches are very similar (similarity > 0.8)")
        print("   The feature projection may be over-smoothing spatial information.")
    else:
        print("\n✅ OK: Patches have sufficient diversity")
    
    # スロット初期化の確認
    print("\n🔍 Slot initialization (mu, sigma):")
    print(f"  slots_mu shape: {model.slot_attention.slots_mu.shape}")
    print(f"  slots_mu mean: {model.slot_attention.slots_mu.mean().item():.6f}")
    print(f"  slots_mu std: {model.slot_attention.slots_mu.std().item():.6f}")
    print(f"  slots_mu min: {model.slot_attention.slots_mu.min().item():.6f}")
    print(f"  slots_mu max: {model.slot_attention.slots_mu.max().item():.6f}")
    
    print(f"\n  slots_log_sigma shape: {model.slot_attention.slots_log_sigma.shape}")
    print(f"  slots_log_sigma mean: {model.slot_attention.slots_log_sigma.mean().item():.6f}")
    print(f"  slots_log_sigma std: {model.slot_attention.slots_log_sigma.std().item():.6f}")
    
    # 実際にサンプルされるスロットの分散
    mu = model.slot_attention.slots_mu  # (1, 5, 64)
    sigma = model.slot_attention.slots_log_sigma.exp()  # (1, 1, 64)
    
    print(f"\n  sigma (exp of log_sigma):")
    print(f"    Mean: {sigma.mean().item():.6f}")
    print(f"    Std: {sigma.std().item():.6f}")
    print(f"    Min: {sigma.min().item():.6f}")
    print(f"    Max: {sigma.max().item():.6f}")
    
    # muの各スロットがどれくらい異なるか
    mu_flat = mu[0]  # (5, 64)
    mu_norm = torch.nn.functional.normalize(mu_flat, dim=1)
    mu_similarity = torch.mm(mu_norm, mu_norm.t())
    mask5 = torch.eye(5) == 0
    mu_off_diag = mu_similarity[mask5]
    
    print(f"\n📊 Slot mu similarity (how different are initial slot centers?):")
    print(f"  Mean: {mu_off_diag.mean().item():.6f}")
    print(f"  Std: {mu_off_diag.std().item():.6f}")
    print(f"  Min: {mu_off_diag.min().item():.6f}")
    print(f"  Max: {mu_off_diag.max().item():.6f}")
    
    if mu_off_diag.mean().item() > 0.9:
        print("\n❌ PROBLEM: All slot mu are nearly identical (similarity > 0.9)")
        print("   Xavier initialization failed or was overridden!")
    else:
        print("\n✅ OK: Slot mu have sufficient diversity")
