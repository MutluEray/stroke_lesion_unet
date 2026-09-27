"""
Day 5 — evaluation against held-out test set and published baselines.

Owns:
    - Full-volume inference via TorchIO's GridSampler/GridAggregator
      (sliding-window with overlap-averaging) — the actual number that's
      comparable to published ISLES'22 baselines, unlike train.py's
      patch-level validation Dice.
    - Per-case and aggregate metrics: Dice, IoU, sensitivity, precision.
    - Writing results/metrics/dice_scores.csv (per-case) and
      results/metrics/results_table.md (aggregate, including the
      published-baseline comparison row for the README/write-up).

Usage:
    python evaluate.py [--checkpoint checkpoints/best.pt]

Outputs:
    - results/metrics/dice_scores.csv
    - results/metrics/results_table.md
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torchio as tio

from dataset import load_case_subject
from model import UNet3D
from utils import load_config, get_device, dice_score, iou_score, sensitivity, precision


def load_model(checkpoint_path, cfg, device):
    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        depth=cfg["model"]["depth"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_dice = checkpoint.get("val_dice", float("nan"))
    print(
        f"Loaded checkpoint from epoch {epoch} "
        f"(patch-level val Dice at save time: {val_dice:.4f} — not the number we're about to compute)"
    )
    return model, epoch


def predict_volume(model, subject, patch_size, overlap_fraction, batch_size, device):
    """Full-volume sliding-window inference: tile the volume into
    overlapping patches, run each through the model, and blend overlapping
    predictions by averaging (TorchIO's GridSampler/GridAggregator)."""
    patch_overlap = tuple(int(round(p * overlap_fraction)) for p in patch_size)
    patch_overlap = tuple(o + (o % 2) for o in patch_overlap)  # GridSampler requires even overlap

    grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")

    with torch.no_grad():
        for patches_batch in patch_loader:
            images = patches_batch["image"][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            logits = model(images)
            probs = torch.sigmoid(logits)
            aggregator.add_batch(probs, locations)

    return aggregator.get_output_tensor()  # (C, X, Y, Z)


def evaluate_case(model, processed_dir, case_id, patch_size, overlap_fraction, batch_size, device, threshold=0.5):
    case_dir = Path(processed_dir) / case_id
    subject = load_case_subject(case_dir, min_size=patch_size)

    pred_probs = predict_volume(model, subject, patch_size, overlap_fraction, batch_size, device)
    pred_probs = pred_probs.unsqueeze(0)                                    # (1, 1, X, Y, Z)
    target = subject["mask"].data.unsqueeze(0).to(pred_probs.device)        # (1, 1, X, Y, Z)

    return {
        "case_id": case_id,
        "dice": dice_score(pred_probs, target, threshold=threshold),
        "iou": iou_score(pred_probs, target, threshold=threshold),
        "sensitivity": sensitivity(pred_probs, target, threshold=threshold),
        "precision": precision(pred_probs, target, threshold=threshold),
        "lesion_voxels_true": int(target.sum().item()),
        "lesion_voxels_pred": int((pred_probs > threshold).float().sum().item()),
    }


def run_evaluation(model, cfg, device):
    processed_dir = cfg["data"]["processed_dir"]
    with open(Path(processed_dir) / "splits.json") as f:
        test_ids = json.load(f)["test"]

    patch_size = tuple(cfg["patching"]["patch_size"])
    overlap_fraction = cfg["evaluation"]["sliding_window_overlap"]
    batch_size = cfg["training"]["batch_size"]

    results = []
    for i, case_id in enumerate(test_ids, 1):
        try:
            result = evaluate_case(
                model, processed_dir, case_id, patch_size, overlap_fraction, batch_size, device
            )
            results.append(result)
            print(f"  [{i}/{len(test_ids)}] {case_id}: dice={result['dice']:.4f} iou={result['iou']:.4f}")
        except Exception as e:
            print(f"  [{i}/{len(test_ids)}] {case_id}: FAILED ({e})")

    return results


def summarize_and_save(results, out_dir, checkpoint_epoch):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No successful evaluations — nothing to summarize.")
        return

    csv_path = out_dir / "dice_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved per-case metrics to {csv_path}")

    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    sens = [r["sensitivity"] for r in results]
    precs = [r["precision"] for r in results]

    def line(name, values):
        return (
            f"{name}: mean={statistics.mean(values):.4f}  median={statistics.median(values):.4f}  "
            f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}"
        )

    print(f"\n{len(results)} test cases evaluated (checkpoint epoch {checkpoint_epoch}, full-volume sliding-window)")
    print(line("Dice", dices))
    print(line("IoU", ious))
    print(line("Sensitivity", sens))
    print(line("Precision", precs))

    table_path = out_dir / "results_table.md"
    with open(table_path, "w") as f:
        f.write(f"# Results — ISLES 2022 held-out test split (n={len(results)})\n\n")
        f.write(
            "Full-volume sliding-window Dice/IoU/sensitivity/precision on this project's "
            "own test split (not the withheld ISLES'22 challenge test set, which has no "
            "public labels).\n\n"
        )
        f.write("| Model | Dice | IoU | Sensitivity | Precision | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(
            f"| This project (checkpoint epoch {checkpoint_epoch}) | "
            f"{statistics.mean(dices):.3f} ± {statistics.pstdev(dices):.3f} | "
            f"{statistics.mean(ious):.3f} ± {statistics.pstdev(ious):.3f} | "
            f"{statistics.mean(sens):.3f} ± {statistics.pstdev(sens):.3f} | "
            f"{statistics.mean(precs):.3f} ± {statistics.pstdev(precs):.3f} | "
            f"single model, reduced 3D U-Net, DWI+ADC only, {len(results)} test cases |\n"
        )
        f.write(
            "| ISLES'22 top team (SegResNet ensemble) | 0.824 | — | — | — | "
            "published, 15-model cross-validated ensemble — not directly comparable "
            "(different test set, far more compute) but the reference point |\n"
        )
    print(f"Saved comparison table to {table_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: config.yaml's checkpoint_dir/best.pt)",
    )
    args = parser.parse_args()

    cfg = load_config()
    device = get_device(cfg)
    print(f"Evaluating on device: {device}")

    checkpoint_path = args.checkpoint or str(Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path} — run train.py first.")

    model, checkpoint_epoch = load_model(checkpoint_path, cfg, device)

    print("\nRunning full-volume sliding-window inference on the test split...")
    results = run_evaluation(model, cfg, device)

    summarize_and_save(results, cfg["evaluation"]["results_dir"], checkpoint_epoch)


if __name__ == "__main__":
    main()
