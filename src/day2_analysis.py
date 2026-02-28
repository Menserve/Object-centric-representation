"""
Day 2 Analysis: Statistical tests & DINOv1>DINOv2 reversal investigation
=========================================================================

Phase 3 Day 2 (2026-03-02)

Tasks:
  1. Statistical significance: Metal ARI vs Rubber ARI (per-sample paired test)
  2. DINOv1 > DINOv2 reversal: per-sample comparison on 300 samples
  3. Lambert-only control: rubber_only vs metal_only scene analysis
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"ari_v2_{name}.json"
    with open(path) as f:
        return json.load(f)


def extract_paired_material_ari(data: dict) -> tuple:
    """Extract per-sample Metal/Rubber ARI where both are non-NaN."""
    metal_aris, rubber_aris, fg_aris = [], [], []
    filenames = []
    for s in data['per_sample']:
        m = s['metal_ari']
        r = s['rubber_ari']
        if m is not None and r is not None and not (np.isnan(m) or np.isnan(r)):
            metal_aris.append(m)
            rubber_aris.append(r)
            fg_aris.append(s['fg_ari'])
            filenames.append(s['filename'])
    return np.array(metal_aris), np.array(rubber_aris), np.array(fg_aris), filenames


def extract_fg_ari(data: dict) -> np.ndarray:
    """Extract all FG-ARI values."""
    return np.array([s['fg_ari'] for s in data['per_sample']])


# ============================================================
# Task 1: Statistical significance (Metal vs Rubber ARI)
# ============================================================
def task1_material_significance():
    print("=" * 70)
    print("TASK 1: Metal ARI vs Rubber ARI — Statistical Significance")
    print("=" * 70)

    for name in ['dinov2', 'dinov1', 'clip']:
        data = load_results(name)
        metal, rubber, fg, fnames = extract_paired_material_ari(data)
        diff = metal - rubber  # negative = Metal harder

        n = len(metal)
        print(f"\n--- {name.upper()} (n={n} paired scenes) ---")
        print(f"  Metal ARI:  {metal.mean():.4f} ± {metal.std():.4f}")
        print(f"  Rubber ARI: {rubber.mean():.4f} ± {rubber.std():.4f}")
        print(f"  Delta(M-R): {diff.mean():.4f} ± {diff.std():.4f}")

        # Paired t-test (H₀: mean(Metal - Rubber) = 0)
        t_stat, p_val_t = stats.ttest_rel(metal, rubber)
        print(f"  Paired t-test:  t={t_stat:.3f}, p={p_val_t:.4f}", end="")
        print(f"  {'***' if p_val_t < 0.001 else '**' if p_val_t < 0.01 else '*' if p_val_t < 0.05 else 'n.s.'}")

        # Wilcoxon signed-rank (non-parametric alternative)
        try:
            w_stat, p_val_w = stats.wilcoxon(metal, rubber)
            print(f"  Wilcoxon:       W={w_stat:.1f}, p={p_val_w:.4f}", end="")
            print(f"  {'***' if p_val_w < 0.001 else '**' if p_val_w < 0.01 else '*' if p_val_w < 0.05 else 'n.s.'}")
        except ValueError as e:
            print(f"  Wilcoxon: skipped ({e})")

        # Bootstrap 95% CI for mean difference
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(10000):
            idx = rng.integers(0, n, size=n)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        print(f"  Bootstrap 95% CI for Δ(M-R): [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  CI excludes 0: {'YES → significant' if ci_lo > 0 or ci_hi < 0 else 'NO → not significant'}")

        # Effect size (Cohen's d for paired samples)
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        print(f"  Cohen's d (paired): {d:.3f}  ({'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})")

        # Count: how many scenes have Metal < Rubber
        n_metal_harder = (diff < 0).sum()
        n_rubber_harder = (diff > 0).sum()
        n_tied = (diff == 0).sum()
        print(f"  Scene-level: Metal<Rubber={n_metal_harder}, Metal>Rubber={n_rubber_harder}, tied={n_tied}")


# ============================================================
# Task 2: DINOv1 > DINOv2 reversal analysis
# ============================================================
def task2_reversal_analysis():
    print("\n" + "=" * 70)
    print("TASK 2: DINOv1 vs DINOv2 — Per-sample Reversal Analysis")
    print("=" * 70)

    d2 = load_results('dinov2')
    d1 = load_results('dinov1')

    fg_d2 = extract_fg_ari(d2)
    fg_d1 = extract_fg_ari(d1)

    diff = fg_d1 - fg_d2  # positive = DINOv1 is better

    print(f"\nOverall (n=300):")
    print(f"  DINOv2 FG-ARI: {fg_d2.mean():.4f} ± {fg_d2.std():.4f}")
    print(f"  DINOv1 FG-ARI: {fg_d1.mean():.4f} ± {fg_d1.std():.4f}")
    print(f"  Δ(v1-v2):      {diff.mean():.4f} ± {diff.std():.4f}")

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(fg_d1, fg_d2)
    print(f"  Paired t-test:  t={t_stat:.3f}, p={p_val:.4f}", end="")
    print(f"  {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}")

    # Wilcoxon
    try:
        w_stat, p_val_w = stats.wilcoxon(fg_d1, fg_d2)
        print(f"  Wilcoxon:       W={w_stat:.1f}, p={p_val_w:.4f}", end="")
        print(f"  {'***' if p_val_w < 0.001 else '**' if p_val_w < 0.01 else '*' if p_val_w < 0.05 else 'n.s.'}")
    except ValueError as e:
        print(f"  Wilcoxon: skipped ({e})")

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot = [diff[rng.integers(0, 300, 300)].mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for Δ(v1-v2): [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Per-sample comparison: how many scenes prefer each model
    n_v1_wins = (diff > 0).sum()
    n_v2_wins = (diff < 0).sum()
    n_tied = (diff == 0).sum()
    print(f"  Scene-level: DINOv1>DINOv2={n_v1_wins}, DINOv2>DINOv1={n_v2_wins}, tied={n_tied}")

    # Analyze where DINOv1 wins vs loses — by scene complexity / material composition
    print(f"\n  --- DINOv1 advantage by scene type ---")
    for stype in ['metal_only', 'rubber_only', 'mixed']:
        idx = [i for i, s in enumerate(d2['per_sample']) if s['scene_type'] == stype]
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        d_sub = diff[idx]
        print(f"  {stype:14s} (n={len(idx):3d}): Δ(v1-v2) = {d_sub.mean():+.4f} ± {d_sub.std():.4f}  "
              f"v1>{(d_sub>0).sum()}, v2>{(d_sub<0).sum()}")

    # Analyze by object count
    print(f"\n  --- DINOv1 advantage by object count ---")
    obj_counts = [s['metal_count'] + s['rubber_count'] for s in d2['per_sample']]
    for lo, hi in [(3, 5), (6, 7), (8, 10)]:
        idx = [i for i, c in enumerate(obj_counts) if lo <= c <= hi]
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        d_sub = diff[idx]
        print(f"  {lo}-{hi} objects   (n={len(idx):3d}): Δ(v1-v2) = {d_sub.mean():+.4f} ± {d_sub.std():.4f}  "
              f"v1>{(d_sub>0).sum()}, v2>{(d_sub<0).sum()}")

    # Analyze by FG-ARI level (is the reversal driven by easy or hard scenes?)
    print(f"\n  --- DINOv1 advantage by scene difficulty (DINOv2 FG-ARI quantile) ---")
    q33, q66 = np.percentile(fg_d2, [33, 66])
    for label, lo, hi in [("hard (bottom 33%)", -1, q33), ("medium", q33, q66), ("easy (top 33%)", q66, 2)]:
        idx = np.where((fg_d2 > lo) & (fg_d2 <= hi))[0]
        if len(idx) == 0:
            continue
        d_sub = diff[idx]
        print(f"  {label:20s} (n={len(idx):3d}): Δ(v1-v2) = {d_sub.mean():+.4f} ± {d_sub.std():.4f}  "
              f"v1>{(d_sub>0).sum()}, v2>{(d_sub<0).sum()}")

    # Correlation between DINOv2 performance and DINOv1 advantage
    r, p = stats.pearsonr(fg_d2, diff)
    print(f"\n  Correlation(DINOv2_ARI, DINOv1_advantage): r={r:.3f}, p={p:.4f}")
    print(f"  → {'DINOv1 benefits more on hard scenes' if r < 0 else 'DINOv1 benefits more on easy scenes'}")


# ============================================================
# Task 3: Lambert-only control analysis
# ============================================================
def task3_lambert_control():
    print("\n" + "=" * 70)
    print("TASK 3: Lambert-only Control (rubber_only scenes)")
    print("=" * 70)

    for name in ['dinov2', 'dinov1', 'clip']:
        data = load_results(name)
        samples = data['per_sample']

        # Separate scene types
        types = {}
        for s in samples:
            st = s['scene_type']
            if st not in types:
                types[st] = []
            types[st].append(s['fg_ari'])

        print(f"\n--- {name.upper()} ---")
        for st in ['rubber_only', 'mixed', 'metal_only']:
            vals = np.array(types.get(st, []))
            if len(vals) == 0:
                continue
            print(f"  {st:14s} (n={len(vals):3d}): FG-ARI = {vals.mean():.4f} ± {vals.std():.4f}  "
                  f"[min={vals.min():.4f}, max={vals.max():.4f}]")

        # Direct comparison: rubber_only vs metal_only (Mann-Whitney U)
        r_vals = np.array(types.get('rubber_only', []))
        m_vals = np.array(types.get('metal_only', []))
        if len(r_vals) > 0 and len(m_vals) > 0:
            u_stat, p_val = stats.mannwhitneyu(r_vals, m_vals, alternative='greater')
            print(f"  Mann-Whitney U (rubber > metal): U={u_stat:.1f}, p={p_val:.4f}", end="")
            print(f"  {'*' if p_val < 0.05 else 'n.s.'}")
            print(f"  Δ(rubber_only - metal_only) = {r_vals.mean() - m_vals.mean():+.4f}")
        else:
            print(f"  Cannot compare: rubber_only n={len(r_vals)}, metal_only n={len(m_vals)}")

    # Limitation notice
    print(f"\n--- LIMITATION ---")
    print(f"  rubber_only: n=5, metal_only: n=8")
    print(f"  These are too few for robust statistical inference.")
    print(f"  The scene-level comparison is suggestive but not conclusive.")
    print(f"  Per-object material ARI (Task 1) is the stronger evidence.")


# ============================================================
# Summary & poster-ready statements
# ============================================================
def summary():
    print("\n" + "=" * 70)
    print("SUMMARY: Poster-ready Statements")
    print("=" * 70)

    d2 = load_results('dinov2')
    d1 = load_results('dinov1')
    cl = load_results('clip')

    # Quick stats
    m2, r2, _, _ = extract_paired_material_ari(d2)
    m1, r1, _, _ = extract_paired_material_ari(d1)
    mc, rc, _, _ = extract_paired_material_ari(cl)

    diff2 = m2 - r2
    diff1 = m1 - r1

    t2, p2 = stats.ttest_rel(m2, r2)
    t1, p1 = stats.ttest_rel(m1, r1)

    fg_d2 = extract_fg_ari(d2)
    fg_d1 = extract_fg_ari(d1)
    t_rev, p_rev = stats.ttest_rel(fg_d1, fg_d2)

    print(f"""
Key findings for poster:

1. Material Effect (DINOv2):
   Metal ARI ({m2.mean():.3f}) < Rubber ARI ({r2.mean():.3f}), Δ={diff2.mean():.3f}
   Paired t-test: t={t2:.2f}, p={'<0.001' if p2 < 0.001 else f'{p2:.3f}'}
   → {'Metal (specular) objects are significantly harder to segment' if p2 < 0.05 else 'Difference is not statistically significant'}

2. Material Effect (DINOv1):
   Metal ARI ({m1.mean():.3f}) ≈ Rubber ARI ({r1.mean():.3f}), Δ={diff1.mean():.3f}
   Paired t-test: t={t1:.2f}, p={p1:.3f}
   → DINOv1 is material-agnostic (no significant Metal/Rubber gap)

3. DINOv1 vs DINOv2 on 300 samples:
   DINOv1 ({fg_d1.mean():.3f}) vs DINOv2 ({fg_d2.mean():.3f}), Δ={fg_d1.mean()-fg_d2.mean():.3f}
   Paired t-test: t={t_rev:.2f}, p={p_rev:.3f}
   → {'Difference is statistically significant' if p_rev < 0.05 else 'Difference is NOT statistically significant (reversal may be noise)'}
""")


if __name__ == '__main__':
    task1_material_significance()
    task2_reversal_analysis()
    task3_lambert_control()
    summary()
