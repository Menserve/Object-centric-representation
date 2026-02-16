#!/usr/bin/env python3
"""
Slot Attentionの各Iteration（1-5）での類似度推移を数値分析
Over-smoothingが本当に起きているかを確認
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train_movi import MoviDataset
from savi_dinosaur import SAViDinosaur


def analyze_iteration_dynamics(checkpoint_path: str, data_dir: str, sample_idx: int = 0):
    """
    各Iterationでのslot間類似度を追跡し、Over-smoothingを検証
    """
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Loaded: Best loss = {checkpoint['loss']:.6f}\n")
    
    # モデルの構築
    model = SAViDinosaur(backbone='dinov2_vits14', num_slots=5, slot_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # データのロード
    dataset = MoviDataset(data_dir, split='test', max_frames=1)
    sample = dataset[sample_idx]
    video = sample['video'].unsqueeze(0)
    
    print(f"Sample: {sample['materials']}\n")
    print("=" * 80)
    
    with torch.no_grad():
        # 特徴量抽出
        features_proj, _ = model.encode(video[:, 0])
        b, n, d = features_proj.shape  # d = slot_dim (e.g., 64)
        features_proj = model.slot_attention.norm_features(features_proj)
        
        # 初期スロット
        slots = model.slot_attention.slots_mu.expand(b, -1, -1)
        
        # Key, Value（全Iterで共通）
        k = model.slot_attention.to_k(features_proj)  # (B, N, kvq_dim)
        v = model.slot_attention.to_v(features_proj)  # (B, N, kvq_dim)
        kvq_dim = k.shape[-1]  # Get actual kvq_dim from k
        
        # Scale factor
        scale = model.slot_attention.scale  # Use model's scale (kvq_dim ** -0.5)
        
        # 各Iterationでの統計を記録
        iter_stats = []
        
        for iter_idx in range(5):
            slots_prev = slots
            slots_normalized = model.slot_attention.norm_slots(slots)
            
            # Query
            q = model.slot_attention.to_q(slots_normalized)  # (B, K, kvq_dim)
            
            # Attention weights
            dots = torch.einsum('bid,bjd->bij', q, k) * scale  # (B, K, N)
            attn = dots.softmax(dim=1) + 1e-8  # Softmax over slots
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn_normalized = attn / attn_sum
            
            # Attention mapの統計
            attn_np = attn[0].cpu().numpy()  # (K, N)
            
            # 各スロットの統計
            coverage_stats = []
            max_attn_stats = []
            entropy_stats = []
            
            for slot_idx in range(5):
                attn_slot = attn_np[slot_idx]
                coverage = (attn_slot > 0.1).sum() / len(attn_slot)
                max_attn = attn_slot.max()
                entropy = -np.sum(attn_slot * np.log(attn_slot + 1e-8))
                
                coverage_stats.append(coverage)
                max_attn_stats.append(max_attn)
                entropy_stats.append(entropy)
            
            # Slot間の類似度（コサイン類似度）
            similarities = []
            for i in range(5):
                for j in range(i + 1, 5):
                    attn_i = attn_np[i]
                    attn_j = attn_np[j]
                    sim = np.dot(attn_i, attn_j) / (np.linalg.norm(attn_i) * np.linalg.norm(attn_j) + 1e-8)
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            
            # Slotベクトル自体の類似度
            slots_np = slots[0].cpu().numpy()  # (K, D)
            slot_vec_similarities = []
            for i in range(5):
                for j in range(i + 1, 5):
                    vec_i = slots_np[i]
                    vec_j = slots_np[j]
                    sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
                    slot_vec_similarities.append(sim)
            
            avg_slot_vec_sim = np.mean(slot_vec_similarities)
            
            # 記録
            iter_stats.append({
                'iter': iter_idx + 1,
                'avg_coverage': np.mean(coverage_stats),
                'avg_max_attn': np.mean(max_attn_stats),
                'avg_entropy': np.mean(entropy_stats),
                'attn_similarity_avg': avg_similarity,
                'attn_similarity_max': max_similarity,
                'attn_similarity_min': min_similarity,
                'slot_vec_similarity_avg': avg_slot_vec_sim,
                'coverage_std': np.std(coverage_stats),
                'max_attn_std': np.std(max_attn_stats)
            })
            
            # 表示
            print(f"Iteration {iter_idx + 1}:")
            print(f"  Attention pattern similarity: avg={avg_similarity:.4f}, max={max_similarity:.4f}, min={min_similarity:.4f}")
            print(f"  Slot vector similarity: avg={avg_slot_vec_sim:.4f}")
            print(f"  Coverage: avg={np.mean(coverage_stats):.1%} ± {np.std(coverage_stats):.1%}")
            print(f"  Max attention: avg={np.mean(max_attn_stats):.4f} ± {np.std(max_attn_stats):.4f}")
            print(f"  Entropy: avg={np.mean(entropy_stats):.2f} ± {np.std(entropy_stats):.2f}")
            
            # GRU更新
            # attn_normalized: (B, K, N), v: (B, N, kvq_dim)
            updates = torch.einsum('bjd,bij->bid', v, attn_normalized)  # (B, K, kvq_dim)
            
            # GRU: kvq_dim → slot_dim
            slots = model.slot_attention.gru(
                updates.reshape(-1, kvq_dim),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + model.slot_attention.mlp(model.slot_attention.norm_pre_ff(slots))
            
            print()
    
    # 推移グラフ
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    iters = [s['iter'] for s in iter_stats]
    
    # Attention pattern類似度
    axes[0, 0].plot(iters, [s['attn_similarity_avg'] for s in iter_stats], 'o-', label='Average', linewidth=2)
    axes[0, 0].plot(iters, [s['attn_similarity_max'] for s in iter_stats], 's--', label='Max', linewidth=2)
    axes[0, 0].plot(iters, [s['attn_similarity_min'] for s in iter_stats], '^--', label='Min', linewidth=2)
    axes[0, 0].axhline(y=0.5, color='red', linestyle=':', label='Threshold (0.5)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Attention Pattern Similarity')
    axes[0, 0].set_title('Attention Pattern Similarity (Cosine)\nHigher = More collapsed')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Slot vector類似度
    axes[0, 1].plot(iters, [s['slot_vec_similarity_avg'] for s in iter_stats], 'o-', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='red', linestyle=':', label='Threshold (0.5)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Slot Vector Similarity')
    axes[0, 1].set_title('Slot Vector Similarity (Cosine)\nHigher = Slots becoming identical')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Coverage
    axes[0, 2].plot(iters, [s['avg_coverage'] for s in iter_stats], 'o-', linewidth=2)
    axes[0, 2].fill_between(
        iters,
        [s['avg_coverage'] - s['coverage_std'] for s in iter_stats],
        [s['avg_coverage'] + s['coverage_std'] for s in iter_stats],
        alpha=0.3
    )
    axes[0, 2].axhline(y=1.0, color='red', linestyle=':', label='Full coverage (100%)')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Coverage (>0.1)')
    axes[0, 2].set_title('Attention Coverage\nLower = More selective')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.05])
    
    # Max attention
    axes[1, 0].plot(iters, [s['avg_max_attn'] for s in iter_stats], 'o-', linewidth=2)
    axes[1, 0].fill_between(
        iters,
        [s['avg_max_attn'] - s['max_attn_std'] for s in iter_stats],
        [s['avg_max_attn'] + s['max_attn_std'] for s in iter_stats],
        alpha=0.3
    )
    axes[1, 0].axhline(y=0.2, color='red', linestyle=':', label='Uniform (1/5 = 0.2)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Max Attention')
    axes[1, 0].set_title('Max Attention per Slot\nHigher = More focused')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(iters, [s['avg_entropy'] for s in iter_stats], 'o-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Attention Entropy\nHigher = More uniform distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 診断テキスト
    axes[1, 2].axis('off')
    diagnosis_text = "Diagnosis:\n\n"
    
    iter1 = iter_stats[0]
    iter5 = iter_stats[-1]
    
    diagnosis_text += f"Iter 1:\n"
    diagnosis_text += f"  Attn sim: {iter1['attn_similarity_avg']:.4f}\n"
    diagnosis_text += f"  Coverage: {iter1['avg_coverage']:.1%}\n\n"
    
    diagnosis_text += f"Iter 5:\n"
    diagnosis_text += f"  Attn sim: {iter5['attn_similarity_avg']:.4f}\n"
    diagnosis_text += f"  Coverage: {iter5['avg_coverage']:.1%}\n\n"
    
    if iter5['attn_similarity_avg'] > 0.95:
        diagnosis_text += "❌ Complete collapse\n"
        diagnosis_text += "(All slots identical)\n\n"
    elif iter5['attn_similarity_avg'] > 0.7:
        diagnosis_text += "⚠️ High similarity\n"
        diagnosis_text += "(Over-smoothing likely)\n\n"
    else:
        diagnosis_text += "✅ Diversity maintained\n\n"
    
    if iter5['attn_similarity_avg'] > iter1['attn_similarity_avg'] + 0.1:
        diagnosis_text += "📈 Over-smoothing detected!\n"
        diagnosis_text += f"Similarity increased:\n"
        diagnosis_text += f"{iter1['attn_similarity_avg']:.4f} → {iter5['attn_similarity_avg']:.4f}\n\n"
        diagnosis_text += "Possible causes:\n"
        diagnosis_text += "- GRU pushing to mean\n"
        diagnosis_text += "- Softmax scale too small\n"
        diagnosis_text += "- Weak initialization\n"
    else:
        diagnosis_text += "📊 Similarity stable\n"
        diagnosis_text += "Problem likely in initialization\n"
    
    axes[1, 2].text(0.1, 0.5, diagnosis_text, fontsize=10, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    save_path = Path(checkpoint_path).parent / 'iteration_dynamics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {save_path}")
    
    # 結論
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if iter5['attn_similarity_avg'] > iter1['attn_similarity_avg'] + 0.1:
        print("🔴 Gemini is CORRECT: Over-smoothing is happening!")
        print(f"   Attention similarity: {iter1['attn_similarity_avg']:.4f} (Iter 1) → {iter5['attn_similarity_avg']:.4f} (Iter 5)")
        print("   The iterative update loop is washing out information.")
    else:
        print("🟡 Alternative hypothesis: Similarity is high from the start.")
        print(f"   Attention similarity: {iter1['attn_similarity_avg']:.4f} (Iter 1) → {iter5['attn_similarity_avg']:.4f} (Iter 5)")
        print("   Problem is likely in slot initialization or softmax scale.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/single_frame_spatial/dinov2_vits14/best_model.pt')
    parser.add_argument('--sample_path', type=str, default='../data/movi_a_subset/metal_000.pt')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    # Extract data_dir from sample_path
    data_dir = str(Path(args.sample_path).parent)
    checkpoint_path = args.checkpoint
    
    analyze_iteration_dynamics(checkpoint_path, data_dir, sample_idx=0)
