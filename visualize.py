"""
Day 6 — portfolio figures.

Owns:
    - 2D slice overlay figures: DWI / ground truth / prediction, side by
      side, for a spread of test cases from results/metrics/dice_scores.csv
      — deliberately worst-to-best, including a failure case, not just the
      best results. Cases with an empty ground-truth mask are excluded
      from selection (trivial Dice=1.0 there — see select_example_cases).
      Each figure uses the axial slice with the most ground-truth lesion
      voxels, not a fixed mid-slice, so small lesions aren't missed — and
      is cropped to the lesion region with a margin (lesion_crop_bounds),
      since a small lesion is otherwise an invisible handful of pixels in
      a full-brain-width panel, making low-Dice failure cases visually
      illegible even though the number is correct.
    - A results summary figure: per-case Dice distribution (histogram)
      alongside the published-baseline comparison bar, for the README.

    No interactive viewer — see the original project plan's reasoning:
    disproportionate build time for a 1-week timeline relative to what it
    adds over good static figures.

Usage:
    python visualize.py [--checkpoint checkpoints/best.pt]

Outputs:
    - results/figures/overlay_<case_id>.png (one per selected case)
    - results/figures/results_comparison.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_case_subject
from evaluate import load_model, predict_volume
from utils import load_config, get_device


def load_dice_scores(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["dice"] = float(r["dice"])
        r["iou"] = float(r["iou"])
        r["lesion_voxels_true"] = int(r["lesion_voxels_true"])
    return rows


def select_example_cases(rows, n_cases):
    """Pick a spread of cases across the performance range — worst, mid,
    best — including failures, not just best-of. Cases with an empty
    ground-truth mask are excluded: their Dice is trivially 1.0 (nothing
    predicted where nothing exists), which would misrepresent the model
    if featured as a "best case"."""
    non_trivial = [r for r in rows if r["lesion_voxels_true"] > 0]
    excluded = len(rows) - len(non_trivial)
    if excluded:
        print(
            f"Excluded {excluded} empty-ground-truth-mask case(s) from example "
            f"selection (trivial Dice=1.0, not informative as a featured example)"
        )

    ranked = sorted(non_trivial, key=lambda r: r["dice"])
    n = len(ranked)
    if n <= n_cases:
        return ranked

    indices = sorted(set(round(i * (n - 1) / (n_cases - 1)) for i in range(n_cases)))
    return [ranked[i] for i in indices]


def best_slice_index(mask, axis=2):
    """Axial slice with the most ground-truth lesion voxels — more
    informative than a fixed mid-slice, especially for small lesions a
    mid-slice could miss entirely."""
    other_axes = tuple(a for a in range(mask.ndim) if a != axis)
    counts = mask.sum(axis=other_axes)
    return int(np.argmax(counts))


def lesion_crop_bounds(mask_slice, pred_slice, margin=15, min_size=40):
    """Bounding box around the union of GT and predicted lesion in this
    slice, with a margin — so small lesions (the ones driving most of the
    low-Dice cases) are actually visible in the figure instead of lost in
    a full-brain-width panel. Falls back to None (full slice) if there's
    no lesion in either mask."""
    union = (mask_slice > 0) | (pred_slice > 0)
    if not union.any():
        return None
    rows, cols = np.where(union)
    r0, r1 = rows.min() - margin, rows.max() + margin
    c0, c1 = cols.min() - margin, cols.max() + margin
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(mask_slice.shape[0], r1), min(mask_slice.shape[1], c1)

    if r1 - r0 < min_size:  # keep a minimum crop size for context even on tiny lesions
        pad = (min_size - (r1 - r0)) // 2
        r0, r1 = max(0, r0 - pad), min(mask_slice.shape[0], r1 + pad)
    if c1 - c0 < min_size:
        pad = (min_size - (c1 - c0)) // 2
        c0, c1 = max(0, c0 - pad), min(mask_slice.shape[1], c1 + pad)
    return r0, r1, c0, c1


def plot_case_overlay(case_id, dwi, mask, pred, dice, out_path):
    slice_idx = best_slice_index(mask)
    dwi_slice = dwi[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]

    crop = lesion_crop_bounds(mask_slice, pred_slice)
    zoomed = crop is not None
    if zoomed:
        r0, r1, c0, c1 = crop
        dwi_slice = dwi_slice[r0:r1, c0:c1]
        mask_slice = mask_slice[r0:r1, c0:c1]
        pred_slice = pred_slice[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(dwi_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("DWI")

    axes[1].imshow(dwi_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(mask_slice.T == 0, mask_slice.T), cmap="Greens", alpha=0.6)
    axes[1].set_title("Ground truth")

    axes[2].imshow(dwi_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(pred_slice.T == 0, pred_slice.T), cmap="Reds", alpha=0.6)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    zoom_note = " (zoomed to lesion region)" if zoomed else ""
    fig.suptitle(f"{case_id} — Dice {dice:.3f}{zoom_note}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_results_summary(rows, out_path, baseline_dice=0.824):
    dices = [r["dice"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(dices, bins=15, color="#4C72B0", edgecolor="white")
    axes[0].axvline(float(np.mean(dices)), color="black", linestyle="--", label=f"mean {np.mean(dices):.3f}")
    axes[0].set_xlabel("Dice")
    axes[0].set_ylabel("Test cases")
    axes[0].set_title(f"Per-case Dice distribution (test set, n={len(rows)})")
    axes[0].legend()

    values = [float(np.mean(dices)), baseline_dice]
    axes[1].bar(
        ["This project\n(mean)", "ISLES'22 top team\n(SegResNet ensemble)"],
        values,
        color=["#4C72B0", "#888888"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Reference comparison\n(different test set — not strictly apples-to-apples)")
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    device = get_device(cfg)

    metrics_dir = Path(cfg["evaluation"]["results_dir"])
    csv_path = metrics_dir / "dice_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run evaluate.py first.")
    rows = load_dice_scores(csv_path)

    figures_dir = Path(cfg["visualization"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_cases = cfg["visualization"]["n_example_cases"]
    selected = select_example_cases(rows, n_cases)
    print(
        f"Selected {len(selected)} example case(s), worst -> best: "
        + ", ".join(f"{r['case_id']} ({r['dice']:.3f})" for r in selected)
    )

    checkpoint_path = args.checkpoint or str(Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    model, checkpoint_epoch = load_model(checkpoint_path, cfg, device)

    patch_size = tuple(cfg["patching"]["patch_size"])
    overlap_fraction = cfg["evaluation"]["sliding_window_overlap"]
    batch_size = cfg["training"]["batch_size"]
    processed_dir = cfg["data"]["processed_dir"]

    for r in selected:
        case_id = r["case_id"]
        subject = load_case_subject(Path(processed_dir) / case_id, min_size=patch_size)
        pred_probs = predict_volume(model, subject, patch_size, overlap_fraction, batch_size, device)

        dwi = subject["image"].data[0].numpy()             # channel 0 = DWI
        mask = subject["mask"].data[0].numpy()
        pred = (pred_probs[0].cpu().numpy() > 0.5).astype(np.float32)

        out_path = figures_dir / f"overlay_{case_id}.png"
        plot_case_overlay(case_id, dwi, mask, pred, r["dice"], out_path)
        print(f"  saved {out_path}")

    summary_path = figures_dir / "results_comparison.png"
    plot_results_summary(rows, summary_path)
    print(f"\nSaved results summary to {summary_path}")


if __name__ == "__main__":
    main()
