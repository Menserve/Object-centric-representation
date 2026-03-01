"""
ARI Evaluation v2: Per-object material analysis with correct labels
===================================================================

Phase 3 (2026-03-01): 正しい material_label を使った再評価

主な改善点:
  - per-object ARI: metal オブジェクトのピクセルと rubber オブジェクトのピクセルを
    分離して個別に ARI を計算
  - scene_type 別集計: metal_only / rubber_only / mixed
  - 既存チェックポイントを新データ (data/movi_a_v2) で評価
  - JSON で結果を保存

Usage:
    python src/compute_ari_v2.py \
        --checkpoint checkpoints/dinov2_singleframe_final/dinov2_vits14/best_model.pt \
        --data_dir data/movi_a_v2 \
        --backbone dinov2_vits14 \
        --output results_dinov2.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from savi_dinosaur import SAViDinosaur


def compute_ari(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Compute ARI between flattened label arrays. Returns NaN if insufficient pixels."""
    if len(pred_labels) < 2:
        return float('nan')
    if len(np.unique(gt_labels)) < 2 and len(np.unique(pred_labels)) < 2:
        return float('nan')
    return adjusted_rand_score(gt_labels, pred_labels)


def compute_per_object_material_ari(
    pred_masks: np.ndarray,
    gt_seg: np.ndarray,
    materials: list,
) -> dict:
    """
    Compute ARI separately for metal and rubber object pixels.

    Args:
        pred_masks: (K, H, W) soft slot masks
        gt_seg: (H, W) GT instance segmentation (0=background, 1..N=objects)
        materials: list of N material strings ('metal'/'rubber'), indexed by object order

    Returns:
        dict with fg_ari, metal_ari, rubber_ari, per_material pixel counts
    """
    K, H, W = pred_masks.shape
    pred_hard = pred_masks.argmax(axis=0).flatten()  # (H*W,)
    gt_flat = gt_seg.flatten()  # (H*W,)

    # Instance IDs in GT (excluding background=0)
    instance_ids = np.unique(gt_flat)
    instance_ids = instance_ids[instance_ids > 0]

    # Build material map: instance_id -> material
    # GT segmentation uses 1-indexed IDs, materials list is 0-indexed
    # instance_id 1 -> materials[0], instance_id 2 -> materials[1], etc.
    mat_map = {}
    for i, iid in enumerate(sorted(instance_ids)):
        if i < len(materials):
            mat_map[iid] = materials[i]

    # Create pixel-level material mask
    metal_pixels = np.zeros(H * W, dtype=bool)
    rubber_pixels = np.zeros(H * W, dtype=bool)
    for iid, mat in mat_map.items():
        mask = gt_flat == iid
        if mat == 'metal':
            metal_pixels |= mask
        elif mat == 'rubber':
            rubber_pixels |= mask

    fg_pixels = gt_flat > 0

    # FG-ARI (all foreground)
    fg_ari = float('nan')
    if fg_pixels.sum() >= 2:
        fg_ari = compute_ari(pred_hard[fg_pixels], gt_flat[fg_pixels])

    # Metal-only ARI: how well does the model segment metal objects from each other?
    metal_ari = float('nan')
    if metal_pixels.sum() >= 2:
        metal_gt = gt_flat[metal_pixels]
        metal_pred = pred_hard[metal_pixels]
        if len(np.unique(metal_gt)) >= 2:
            metal_ari = compute_ari(metal_pred, metal_gt)

    # Rubber-only ARI
    rubber_ari = float('nan')
    if rubber_pixels.sum() >= 2:
        rubber_gt = gt_flat[rubber_pixels]
        rubber_pred = pred_hard[rubber_pixels]
        if len(np.unique(rubber_gt)) >= 2:
            rubber_ari = compute_ari(rubber_pred, rubber_gt)

    # Full-ARI (including background)
    full_ari = compute_ari(pred_hard, gt_flat)

    return {
        'fg_ari': fg_ari,
        'full_ari': full_ari,
        'metal_ari': metal_ari,
        'rubber_ari': rubber_ari,
        'n_metal_pixels': int(metal_pixels.sum()),
        'n_rubber_pixels': int(rubber_pixels.sum()),
        'n_fg_pixels': int(fg_pixels.sum()),
        'n_metal_objects': sum(1 for m in mat_map.values() if m == 'metal'),
        'n_rubber_objects': sum(1 for m in mat_map.values() if m == 'rubber'),
    }


def load_model(checkpoint_path: str, backbone: str, mask_temperature: float = 0.5,
               num_slots: int = 7, device: str = 'cuda', image_size: int = 224) -> SAViDinosaur:
    """Load model from checkpoint."""
    print(f"Loading model: {backbone} from {checkpoint_path} (image_size={image_size})")

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Auto-detect num_slots
    if 'slot_attention.slots_mu' in state_dict:
        detected = state_dict['slot_attention.slots_mu'].shape[1]
        if detected != num_slots:
            print(f"  Auto-adjusting num_slots: {num_slots} -> {detected}")
            num_slots = detected

    model = SAViDinosaur(
        num_slots=num_slots,
        backbone=backbone,
        mask_temperature=mask_temperature,
        image_size=image_size
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_sample(path: str, target_size=(224, 224), max_frames=1):
    """Load a single .pt sample and resize."""
    data = torch.load(path, weights_only=False)
    video = data['video']  # (T, 3, H, W)
    seg = data['segmentation']  # (T, H, W)

    if max_frames is not None:
        video = video[:max_frames]
        seg = seg[:max_frames]

    t, c, h, w = video.shape
    if (h, w) != target_size:
        video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)
        seg = F.interpolate(
            seg.unsqueeze(1).float(), size=target_size, mode='nearest'
        ).squeeze(1).long()

    return {
        'video': video,
        'segmentation': seg,
        'materials': data.get('materials', []),
        'scene_type': data.get('scene_type', 'unknown'),
        'has_metal': data.get('has_metal', None),
        'metal_count': data.get('metal_count', 0),
        'rubber_count': data.get('rubber_count', 0),
        'filename': os.path.basename(path),
    }


def evaluate_v2(
    checkpoint_path: str,
    data_dir: str,
    backbone: str = 'dinov2_vits14',
    mask_temperature: float = 0.5,
    num_slots: int = 7,
    max_frames: int = 1,
    loss_type: str = 'mse',
    device: str = 'cuda',
    image_size: int = 224,
) -> dict:
    """
    Evaluate checkpoint on v2 data with per-object material ARI.
    """
    model = load_model(checkpoint_path, backbone, mask_temperature, num_slots, device, image_size=image_size)

    # Find all scene files
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt') and f.startswith('scene_')])
    print(f"Evaluating {len(files)} samples from {data_dir}")

    all_results = []

    with torch.no_grad():
        for i, fname in enumerate(files):
            sample = load_sample(
                os.path.join(data_dir, fname),
                target_size=(image_size, image_size),
                max_frames=max_frames
            )
            video = sample['video'].unsqueeze(0).to(device)  # (1, T, 3, H, W)
            gt_seg = sample['segmentation'].numpy()  # (T, H, W)
            materials = sample['materials']

            result = model.forward_video(video, return_all_masks=True, loss_type=loss_type)
            masks = result['all_masks'][0, :, :, 0].cpu().numpy()  # (T, K, H, W)

            T = masks.shape[0]
            frame_results = []

            for t in range(T):
                pred_mask_t = masks[t]  # (K, H, W)
                gt_seg_t = gt_seg[t]  # (H, W)

                # Resize GT seg to match pred mask
                _, pred_h, pred_w = pred_mask_t.shape
                gt_h, gt_w = gt_seg_t.shape
                if gt_h != pred_h or gt_w != pred_w:
                    gt_seg_t = F.interpolate(
                        torch.from_numpy(gt_seg_t).float().unsqueeze(0).unsqueeze(0),
                        size=(pred_h, pred_w), mode='nearest'
                    ).squeeze().numpy().astype(int)

                fr = compute_per_object_material_ari(pred_mask_t, gt_seg_t, materials)
                frame_results.append(fr)

            # Average over frames
            def safe_mean(vals):
                valid = [v for v in vals if not np.isnan(v)]
                return float(np.mean(valid)) if valid else float('nan')

            sample_result = {
                'filename': fname,
                'scene_type': sample['scene_type'],
                'materials': materials,
                'metal_count': sample['metal_count'],
                'rubber_count': sample['rubber_count'],
                'fg_ari': safe_mean([fr['fg_ari'] for fr in frame_results]),
                'full_ari': safe_mean([fr['full_ari'] for fr in frame_results]),
                'metal_ari': safe_mean([fr['metal_ari'] for fr in frame_results]),
                'rubber_ari': safe_mean([fr['rubber_ari'] for fr in frame_results]),
            }
            all_results.append(sample_result)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(files)}] FG-ARI: {sample_result['fg_ari']:.4f}, "
                      f"Metal: {sample_result['metal_ari']:.4f}, Rubber: {sample_result['rubber_ari']:.4f}")

    # ========== Aggregate ==========
    def agg(key, filter_fn=None):
        vals = [r[key] for r in all_results if (filter_fn is None or filter_fn(r)) and not np.isnan(r[key])]
        if not vals:
            return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}

    summary = {
        'backbone': backbone,
        'loss_type': loss_type,
        'checkpoint': checkpoint_path,
        'data_dir': data_dir,
        'n_samples': len(files),
        # Overall
        'fg_ari': agg('fg_ari'),
        'full_ari': agg('full_ari'),
        # Per-object material ARI (ALL scenes)
        'metal_ari': agg('metal_ari'),
        'rubber_ari': agg('rubber_ari'),
        # Scene-type breakdown
        'metal_only_fg_ari': agg('fg_ari', lambda r: r['scene_type'] == 'metal_only'),
        'rubber_only_fg_ari': agg('fg_ari', lambda r: r['scene_type'] == 'rubber_only'),
        'mixed_fg_ari': agg('fg_ari', lambda r: r['scene_type'] == 'mixed'),
        # Metal ARI in mixed scenes (the most informative)
        'mixed_metal_ari': agg('metal_ari', lambda r: r['scene_type'] == 'mixed'),
        'mixed_rubber_ari': agg('rubber_ari', lambda r: r['scene_type'] == 'mixed'),
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  Evaluation Results: {backbone} ({loss_type})")
    print(f"{'=' * 70}")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  FG-ARI (all):         {summary['fg_ari']['mean']:.4f} ± {summary['fg_ari']['std']:.4f} (n={summary['fg_ari']['n']})")
    print(f"  Full-ARI (all):       {summary['full_ari']['mean']:.4f} ± {summary['full_ari']['std']:.4f}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Metal obj ARI (all):  {summary['metal_ari']['mean']:.4f} ± {summary['metal_ari']['std']:.4f} (n={summary['metal_ari']['n']})")
    print(f"  Rubber obj ARI (all): {summary['rubber_ari']['mean']:.4f} ± {summary['rubber_ari']['std']:.4f} (n={summary['rubber_ari']['n']})")
    print(f"  Delta (Metal-Rubber): {summary['metal_ari']['mean'] - summary['rubber_ari']['mean']:+.4f}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Scene: metal_only FG: {summary['metal_only_fg_ari']['mean']:.4f} (n={summary['metal_only_fg_ari']['n']})")
    print(f"  Scene: rubber_only FG:{summary['rubber_only_fg_ari']['mean']:.4f} (n={summary['rubber_only_fg_ari']['n']})")
    print(f"  Scene: mixed FG:      {summary['mixed_fg_ari']['mean']:.4f} (n={summary['mixed_fg_ari']['n']})")
    print(f"  Mixed: metal obj:     {summary['mixed_metal_ari']['mean']:.4f} (n={summary['mixed_metal_ari']['n']})")
    print(f"  Mixed: rubber obj:    {summary['mixed_rubber_ari']['mean']:.4f} (n={summary['mixed_rubber_ari']['n']})")
    print(f"{'=' * 70}")

    return {
        'summary': summary,
        'per_sample': all_results,
    }


def main():
    parser = argparse.ArgumentParser(description='ARI v2: per-object material evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/movi_a_v2')
    parser.add_argument('--backbone', type=str, default='dinov2_vits14',
                        choices=['dinov2_vits14', 'dino_vits16', 'clip_vitb16'])
    parser.add_argument('--mask_temperature', type=float, default=0.5)
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--max_frames', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'cosine', 'channel_norm'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto-generated)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (224 or 448)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = evaluate_v2(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        backbone=args.backbone,
        mask_temperature=args.mask_temperature,
        num_slots=args.num_slots,
        max_frames=args.max_frames,
        loss_type=args.loss_type,
        device=device,
        image_size=args.image_size,
    )

    # Save results
    output_path = args.output
    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = f"results/ari_v2_{args.backbone}_{args.loss_type}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
