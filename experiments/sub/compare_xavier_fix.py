#!/usr/bin/env python3
"""
Xavier初期化前後の比較分析

修正前: checkpoints/single_frame_spatial (randn * 0.02)
修正後: checkpoints/xavier_init_single_frame (Xavier init)
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train_movi import MoviDataset
from savi_dinosaur import SAViDinosaur


def compare_checkpoints(before_path: str, after_path: str, data_dir: str, sample_idx: int = 0):
    """修正前後のCheckpointを比較"""
    
    # データのロード
    dataset = MoviDataset(data_dir, split='test', max_frames=1)
    sample = dataset[sample_idx]
    video = sample['video'].unsqueeze(0)
    
    results = {}
    
    for name, ckpt_path in [("Before (randn*0.02)", before_path), ("After (Xavier)", after_path)]:
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")
        
        # チェックポイントのロード
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Best loss: {checkpoint['loss']:.6f}")
        
        # モデルの構築
        model = SAViDinosaur(backbone='dinov2_vits14', num_slots=5, slot_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            # 特徴量抽出
            features_proj, _ = model.encode(video[:, 0])
            b, n, d = features_proj.shape
            features_proj = model.slot_attention.norm_features(features_proj)
            
            # 初期スロット
            slots = model.slot_attention.slots_mu.expand(b, -1, -1)
            
            # Key, Value
            k = model.slot_attention.to_k(features_proj)
            v = model.slot_attention.to_v(features_proj)
            
            # 各Iterationの統計
            iter_similarities = []
            
            for iter_idx in range(5):
                slots_prev = slots
                slots_normalized = model.slot_attention.norm_slots(slots)
                q = model.slot_attention.to_q(slots_normalized)
                
                scale = d ** -0.5
                dots = torch.einsum('bid,bjd->bij', q, k) * scale
                attn = dots.softmax(dim=1) + 1e-8
                attn_sum = attn.sum(dim=-1, keepdim=True)
                attn_normalized = attn / attn_sum
                
                # Attention similarity
                attn_np = attn[0].cpu().numpy()
                similarities = []
                for i in range(5):
                    for j in range(i + 1, 5):
                        sim = np.dot(attn_np[i], attn_np[j]) / (
                            np.linalg.norm(attn_np[i]) * np.linalg.norm(attn_np[j]) + 1e-8
                        )
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                iter_similarities.append(avg_sim)
                
                print(f"  Iter {iter_idx + 1}: Similarity = {avg_sim:.4f}")
                
                # GRU更新
                updates = torch.einsum('bjhd,bihj->bihd', 
                                      v.view(b, n, 1, d), 
                                      attn.unsqueeze(2))
                updates = updates.squeeze(2)
                
                slots = model.slot_attention.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d)
                )
                slots = slots.reshape(b, -1, d)
                slots = slots + model.slot_attention.mlp(
                    model.slot_attention.norm_pre_ff(slots)
                )
        
        # 初期化パラメータの統計
        slots_mu = model.slot_attention.slots_mu.data.cpu().numpy()[0]  # (K, D)
        slots_logsigma = model.slot_attention.slots_log_sigma.data.cpu().numpy()[0, 0]  # (D,)
        
        print(f"\nInitialization stats:")
        print(f"  slots_mu: std={slots_mu.std():.4f}, mean={slots_mu.mean():.4f}")
        print(f"  slots_logsigma: std={slots_logsigma.std():.4f}, mean={slots_logsigma.mean():.4f}")
        print(f"  sigma (exp): mean={np.exp(slots_logsigma).mean():.4f}")
        
        results[name] = {
            'iter_similarities': iter_similarities,
            'final_loss': checkpoint['loss'],
            'slots_mu_std': slots_mu.std(),
            'sigma_mean': np.exp(slots_logsigma).mean()
        }
    
    # 比較グラフ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Iteration similarity推移
    for name, data in results.items():
        iters = list(range(1, 6))
        axes[0].plot(iters, data['iter_similarities'], 'o-', 
                    label=name, linewidth=2, markersize=8)
    
    axes[0].axhline(y=0.5, color='red', linestyle=':', label='Threshold (0.5)', linewidth=2)
    axes[0].axhline(y=0.95, color='orange', linestyle=':', label='Collapse (0.95)', linewidth=2)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Attention Pattern Similarity', fontsize=12)
    axes[0].set_title('Over-smoothing Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # 初期化パラメータ比較
    names = list(results.keys())
    mu_stds = [results[name]['slots_mu_std'] for name in names]
    sigma_means = [results[name]['sigma_mean'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, mu_stds, width, label='slots_mu std', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, sigma_means, width, label='sigma mean', alpha=0.8)
    
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Initialization Parameters', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = '/home/menserve/Object-centric-representation/checkpoints/xavier_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison to {save_path}")
    
    # 結論
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    before_final = results["Before (randn*0.02)"]['iter_similarities'][-1]
    after_final = results["After (Xavier)"]['iter_similarities'][-1]
    
    improvement = before_final - after_final
    
    if improvement > 0.05:
        print(f"✅ SIGNIFICANT IMPROVEMENT!")
        print(f"   Final similarity: {before_final:.4f} → {after_final:.4f}")
        print(f"   Improvement: {improvement:.4f}")
        print(f"   Xavier initialization successfully reduced over-smoothing!")
    elif improvement > 0:
        print(f"⚠️ MINOR IMPROVEMENT")
        print(f"   Final similarity: {before_final:.4f} → {after_final:.4f}")
        print(f"   Improvement: {improvement:.4f}")
        print(f"   Additional fixes may be needed.")
    else:
        print(f"❌ NO IMPROVEMENT")
        print(f"   Final similarity: {before_final:.4f} → {after_final:.4f}")
        print(f"   Xavier initialization alone is not sufficient.")
        print(f"   Consider: Multi-head attention, temperature scaling, or stronger diversity loss.")


if __name__ == '__main__':
    before_path = '../checkpoints/single_frame_spatial/dinov2_vits14/best_model.pt'
    after_path = '../checkpoints/xavier_init_single_frame/dinov2_vits14/best_model.pt'
    data_dir = '../data/movi_a_subset'
    
    compare_checkpoints(before_path, after_path, data_dir, sample_idx=0)
