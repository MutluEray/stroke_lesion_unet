"""
Day 1 — dataset exploration + compute go/no-go.

Owns:
    - Scanning cases under data/raw/ISLES-2022, reporting per-case and
      aggregated: volume shape, voxel spacing, DWI/ADC intensity range,
      and lesion volume (from the mask) — implemented below.
    - A dummy-batch timing test at config.yaml's target patch size, on the
      configured device, using a structural proxy of the real 3D U-Net
      (model.py isn't built yet — this stands in for it just to time a
      forward+backward step). This is the number that decides whether Day 3
      proceeds with full 3D or flips patching.mode to "2p5d".

Usage:
    python explore.py

Outputs:
    - Prints a summary + go/no-go verdict to stdout
    - Writes results/metrics/dataset_stats.csv
"""

import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from utils import load_config, find_cases, case_paths, load_volume


# ---------------------------------------------------------------------------
# Per-case stats
# ---------------------------------------------------------------------------


def summarize_case(raw_dir, case_id):
    dwi_path, adc_path, mask_path = case_paths(raw_dir, case_id)

    dwi, spacing = load_volume(dwi_path)
    adc, _ = load_volume(adc_path)
    mask, _ = load_volume(mask_path)

    voxel_volume_mm3 = float(np.prod(spacing))
    lesion_voxels = int((mask > 0).sum())

    return {
        "case_id": case_id,
        "shape": dwi.shape,
        "spacing_mm": tuple(round(float(z), 3) for z in spacing),
        "dwi_p1": float(np.percentile(dwi, 1)),
        "dwi_p99": float(np.percentile(dwi, 99)),
        "adc_p1": float(np.percentile(adc, 1)),
        "adc_p99": float(np.percentile(adc, 99)),
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": round(lesion_voxels * voxel_volume_mm3, 1),
    }


def run_exploration(raw_dir, cases):
    rows = []
    skipped = []
    for i, case_id in enumerate(cases, 1):
        try:
            rows.append(summarize_case(raw_dir, case_id))
        except Exception as e:
            skipped.append((case_id, str(e)))
        if i % 50 == 0 or i == len(cases):
            print(f"  processed {i}/{len(cases)}")

    if skipped:
        print(f"\n{len(skipped)} case(s) skipped (unexpected file layout):")
        for case_id, err in skipped[:5]:
            print(f"  - {case_id}: {err}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return rows


def print_summary(rows):
    if not rows:
        print("No cases summarized — check case_paths() naming against your actual files.")
        return

    shapes = sorted(set(r["shape"] for r in rows))
    spacings = sorted(set(r["spacing_mm"] for r in rows))
    lesion_vols = sorted(r["lesion_volume_mm3"] for r in rows)
    n_zero = sum(1 for v in lesion_vols if v == 0)
    zero_ids = [r["case_id"] for r in rows if r["lesion_volume_mm3"] == 0]

    print(f"\n{len(rows)} cases summarized")
    print(f"Distinct shapes: {len(shapes)} — e.g. {shapes[:3]}")
    print(f"Distinct spacings (mm): {len(spacings)} — e.g. {spacings[:3]}")
    print(
        f"Lesion volume (mm^3): min={lesion_vols[0]:.1f}  "
        f"median={lesion_vols[len(lesion_vols)//2]:.1f}  max={lesion_vols[-1]:.1f}"
    )
    print(f"Cases with zero lesion voxels (mask empty): {n_zero}/{len(rows)} — {zero_ids}")

    # Median spacing per axis, and a concrete target_spacing recommendation.
    x_sp = sorted(r["spacing_mm"][0] for r in rows)
    y_sp = sorted(r["spacing_mm"][1] for r in rows)
    z_sp = sorted(r["spacing_mm"][2] for r in rows)
    median_spacing = [
        round(statistics.median(x_sp), 3),
        round(statistics.median(y_sp), 3),
        round(statistics.median(z_sp), 3),
    ]
    print(f"\nMedian spacing across dataset (mm): {median_spacing}")
    print(
        f"Recommended config.yaml line:\n"
        f"  preprocessing:\n"
        f"    target_spacing: {median_spacing}"
    )
    print(
        "(resampling to the dataset's median spacing, rather than forcing full "
        "1mm isotropic, keeps interpolation modest while still giving every "
        "case a consistent physical patch size — a reasonable default for a "
        "1-week timeline.)"
    )


def save_csv(rows, out_path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-case stats to {out_path}")


# ---------------------------------------------------------------------------
# Compute go/no-go: timing proxy
# ---------------------------------------------------------------------------

class TimingProxyUNet3D(nn.Module):
    """Structural stand-in for the real model in model.py (not built yet) —
    same depth/filter-doubling shape, only for timing a forward+backward
    step before committing Day 3 to full 3D."""

    def __init__(self, in_channels, base_filters, depth):
        super().__init__()
        chs = [in_channels] + [base_filters * (2 ** i) for i in range(depth)]
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(chs[i], chs[i + 1], 3, padding=1),
                nn.InstanceNorm3d(chs[i + 1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(chs[i + 1], chs[i + 1], 3, padding=1),
                nn.InstanceNorm3d(chs[i + 1]),
                nn.LeakyReLU(inplace=True),
            )
            for i in range(depth)
        ])
        self.pool = nn.MaxPool3d(2)
        self.head = nn.Conv3d(chs[-1], 1, 1)

    def forward(self, x):
        for block in self.down:
            x = block(x)
            x = self.pool(x)
        x = nn.functional.interpolate(x, size=(4, 4, 4))
        return self.head(x)


def resolve_device(device_name):
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def time_training_step(cfg, n_train_cases):
    patch = cfg["patching"]["patch_size"]
    in_ch = cfg["model"]["in_channels"]
    base_f = cfg["model"]["base_filters"]
    depth = cfg["model"]["depth"]
    batch = cfg["training"]["batch_size"]

    device = resolve_device(cfg["training"]["device"])
    print(f"\nTiming on device: {device}")

    model = TimingProxyUNet3D(in_ch, base_f, depth).to(device)
    x = torch.randn(batch, in_ch, *patch, device=device)
    target = torch.randint(0, 2, (batch, 1, 4, 4, 4), device=device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def step():
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    for _ in range(2):  # warmup, not timed
        step()

    n_steps = 5
    start = time.time()
    for _ in range(n_steps):
        step()
    elapsed = time.time() - start
    per_step = elapsed / n_steps

    patches_per_epoch = n_train_cases * cfg["patching"]["patches_per_volume"]
    steps_per_epoch = patches_per_epoch / batch
    est_epoch_min = (steps_per_epoch * per_step) / 60

    print(f"Patch size: {patch}, batch size: {batch}, base_filters: {base_f}, depth: {depth}")
    print(f"Avg step time: {per_step:.3f}s")
    print(f"Train cases: {n_train_cases}, est. steps/epoch: {steps_per_epoch:.0f}")
    print(f"Est. epoch time: {est_epoch_min:.1f} min")

    print()
    if est_epoch_min > 15:
        print(">>> GO/NO-GO: this looks too slow for ~50-100 epochs across the remaining days.")
        print('>>> Recommend flipping patching.mode to "2p5d" in config.yaml before Day 3.')
    elif est_epoch_min > 6:
        print(">>> GO/NO-GO: borderline. Feasible but will eat most of Day 3-4's budget.")
        print('>>> Consider a smaller patch_size or base_filters, or accept fewer epochs.')
    else:
        print(">>> GO/NO-GO: full 3D patch training looks feasible on this timeline.")


# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]

    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} cases — summarizing all of them")

    rows = run_exploration(raw_dir, cases)
    print_summary(rows)
    save_csv(rows, Path("results/metrics/dataset_stats.csv"))

    n_train = int(round(len(cases) * cfg["data"]["train_frac"]))
    time_training_step(cfg, n_train_cases=n_train)


if __name__ == "__main__":
    main()
