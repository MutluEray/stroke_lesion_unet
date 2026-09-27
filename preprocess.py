"""
Day 2 — preprocessing pipeline.

Owns:
    - Generating the seeded train/val/test split (uses find_cases from
      utils.py).
    - Cropping each volume to its brain bounding box.
    - Per-volume intensity clipping + z-score normalization, computed over
      foreground (nonzero) voxels only so the large zero background
      doesn't skew the statistics — background stays exactly 0 after.
    - Resampling to config.yaml's `preprocessing.target_spacing` (linear
      interpolation for DWI/ADC, nearest-neighbor for the mask, then
      re-binarized — preserves the binary label instead of blurring it).
    - Writing preprocessed volumes to data/processed/<case_id>/.

Usage:
    python preprocess.py

Outputs:
    - data/processed/splits.json  (train/val/test case ID lists)
    - data/processed/<case_id>/{dwi,adc,mask}.nii.gz
"""

import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from utils import load_config, find_cases, case_paths, load_volume


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def make_splits(cases, train_frac, val_frac, test_frac, seed):
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6, f"train/val/test fractions must sum to 1, got {total}"

    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------

def brain_bbox(reference_volume, margin=2):
    """Bounding box of nonzero voxels in `reference_volume`, with a small
    margin, clipped to array bounds. Applied identically to DWI/ADC/mask so
    they stay aligned."""
    nonzero = np.argwhere(reference_volume > 0)
    if nonzero.size == 0:
        return tuple(slice(0, s) for s in reference_volume.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    return tuple(
        slice(max(0, int(mn) - margin), min(sz, int(mx) + margin))
        for mn, mx, sz in zip(mins, maxs, reference_volume.shape)
    )


def normalize_foreground(volume, clip_percentiles):
    """Clip + z-score normalize using foreground (nonzero) voxel statistics
    only. Background stays exactly 0 after normalization."""
    foreground = volume > 0
    if not foreground.any():
        return volume.astype(np.float32)

    lo, hi = np.percentile(volume[foreground], clip_percentiles)
    clipped = np.clip(volume, lo, hi)

    fg_vals = clipped[foreground]
    mean, std = float(fg_vals.mean()), float(fg_vals.std())
    std = std if std > 1e-6 else 1.0

    normalized = (clipped - mean) / std
    normalized[~foreground] = 0.0
    return normalized.astype(np.float32)


def resample_array(volume, original_spacing, target_spacing, is_mask=False):
    """Resample a numpy volume (nibabel axis order: x, y, z) from
    original_spacing to target_spacing via SimpleITK. Origin/direction are
    left at identity — absolute world position doesn't matter here, only
    each case's internal voxel-to-physical scale."""
    sitk_img = sitk.GetImageFromArray(np.transpose(volume, (2, 1, 0)))  # -> (z, y, x)
    sitk_img.SetSpacing(tuple(float(s) for s in original_spacing))

    original_size = sitk_img.GetSize()
    new_size = [
        max(1, int(round(osz * ospc / tspc)))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    resampled = resampler.Execute(sitk_img)
    return np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))  # back to (x, y, z)


def preprocess_case(raw_dir, case_id, target_spacing, clip_percentiles):
    dwi_path, adc_path, mask_path = case_paths(raw_dir, case_id)

    dwi, spacing = load_volume(dwi_path)
    adc, _ = load_volume(adc_path)
    mask, _ = load_volume(mask_path)

    if not (dwi.shape == adc.shape == mask.shape):
        raise ValueError(
            f"shape mismatch — dwi={dwi.shape}, adc={adc.shape}, mask={mask.shape}"
        )

    bbox = brain_bbox(dwi)
    dwi, adc, mask = dwi[bbox], adc[bbox], mask[bbox]

    dwi = normalize_foreground(dwi, clip_percentiles)
    adc = normalize_foreground(adc, clip_percentiles)

    if target_spacing is not None:
        dwi = resample_array(dwi, spacing, target_spacing, is_mask=False)
        adc = resample_array(adc, spacing, target_spacing, is_mask=False)
        mask = resample_array(mask, spacing, target_spacing, is_mask=True)

    mask = (mask > 0.5).astype(np.float32)  # re-binarize after interpolation
    return dwi, adc, mask


def save_case(out_dir, case_id, dwi, adc, mask, target_spacing):
    case_out = Path(out_dir) / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    spacing = target_spacing if target_spacing is not None else (1.0, 1.0, 1.0)
    affine = np.diag(list(spacing) + [1.0]).astype(np.float64)

    nib.save(nib.Nifti1Image(dwi, affine), case_out / "dwi.nii.gz")
    nib.save(nib.Nifti1Image(adc, affine), case_out / "adc.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), case_out / "mask.nii.gz")


def run_preprocessing(cfg, cases):
    raw_dir = cfg["data"]["raw_dir"]
    out_dir = cfg["data"]["processed_dir"]
    target_spacing = cfg["preprocessing"]["target_spacing"]
    clip_percentiles = cfg["preprocessing"]["intensity_clip_percentiles"]

    failed = []
    for i, case_id in enumerate(cases, 1):
        try:
            dwi, adc, mask = preprocess_case(raw_dir, case_id, target_spacing, clip_percentiles)
            save_case(out_dir, case_id, dwi, adc, mask, target_spacing)
        except Exception as e:
            failed.append((case_id, str(e)))
        if i % 25 == 0 or i == len(cases):
            print(f"  preprocessed {i}/{len(cases)}")

    if failed:
        print(f"\n{len(failed)} case(s) failed preprocessing:")
        for case_id, err in failed[:5]:
            print(f"  - {case_id}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    return len(cases) - len(failed)


def main():
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]

    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} cases under {raw_dir}")

    splits = make_splits(
        cases,
        cfg["data"]["train_frac"],
        cfg["data"]["val_frac"],
        cfg["data"]["test_frac"],
        cfg["data"]["seed"],
    )
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} cases")

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved split to {splits_path}")

    target_spacing = cfg["preprocessing"]["target_spacing"]
    print(f"\nPreprocessing all {len(cases)} cases (target_spacing={target_spacing})...")
    n_ok = run_preprocessing(cfg, cases)
    print(f"\nDone: {n_ok}/{len(cases)} cases preprocessed to {out_dir}")


if __name__ == "__main__":
    main()
