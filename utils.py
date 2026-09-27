"""
Shared helpers used across the project's scripts.

Owns:
    - Config loading (load_config), resolved relative to this file so
      every script works regardless of the caller's cwd.
    - Locating case folders under data/raw/ (find_cases) and resolving
      each case's DWI/ADC/mask file paths (case_paths) — matches this
      project's actual on-disk ISLES-2022 layout (no rawdata/ wrapper;
      DWI+ADC under ses-*/dwi/; mask under derivatives/<case>/ses-*/).
    - Volume loading (load_volume) via nibabel.
    - Reproducibility (seed_everything) and device resolution (get_device),
      with automatic MPS -> CUDA -> CPU fallback.
    - Metric functions: dice_score, iou_score, sensitivity, precision
      (all discrete/thresholded — used by both train.py's validation loop
      and evaluate.py's test-set scoring, so the two report numbers on
      the same definition).

Usage:
    from utils import load_config, find_cases, case_paths, load_volume
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml


def load_config(path=None):
    """Load config.yaml. Defaults to the file sitting next to this script,
    so it works regardless of the caller's current working directory."""
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # PyYAML quirk: scientific notation without an explicit decimal point
    # (e.g. "1e-4") parses as a string, not a float — silently breaks
    # anything doing arithmetic on it. Force the numeric training fields
    # that are most likely to be written that way.
    cfg["training"]["lr"] = float(cfg["training"]["lr"])

    return cfg


def find_cases(raw_dir):
    """Return sorted case IDs (subject folder names) found under raw_dir.

    ISLES-2022 as published follows raw_dir/rawdata/sub-strokecaseXXXX/...,
    but this project's actual extraction drops that rawdata/ wrapper —
    sub-* folders sit directly under raw_dir, with derivatives/ as a
    sibling. This checks for rawdata/ first and falls back to raw_dir
    directly. If neither layout matches, it raises with the path it
    actually looked in, rather than silently finding zero cases and
    letting a downstream script fail confusingly.
    """
    raw_dir = Path(raw_dir)
    rawdata_dir = raw_dir / "rawdata"
    scan_dir = rawdata_dir if rawdata_dir.is_dir() else raw_dir

    if not scan_dir.exists():
        raise FileNotFoundError(
            f"{scan_dir} does not exist. Check that the extracted ISLES-2022 "
            f"data lives at {raw_dir}."
        )

    cases = sorted(
        p.name for p in scan_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )
    if not cases:
        raise FileNotFoundError(
            f"No 'sub-*' case folders found under {scan_dir}. "
            f"Run `find {raw_dir} -maxdepth 3 -type d` and check the actual "
            f"layout against find_cases()."
        )
    return cases


def case_paths(raw_dir, case_id):
    """Resolve DWI/ADC/mask file paths for a case.

    Matches this project's actual on-disk layout: each case folder sits
    directly under raw_dir (no rawdata/ wrapper), DWI+ADC live in a dwi/
    subfolder under the session, and derivatives/<case>/ses-*/ holds the
    mask exactly as published.
    """
    raw_dir = Path(raw_dir)
    case_dir = raw_dir / case_id
    derivatives_case_dir = raw_dir / "derivatives" / case_id

    ses_dirs = sorted(case_dir.glob("ses-*"))
    if not ses_dirs:
        raise FileNotFoundError(f"No ses-* folder under {case_dir}")
    ses = ses_dirs[0].name

    dwi = case_dir / ses / "dwi" / f"{case_id}_{ses}_dwi.nii.gz"
    adc = case_dir / ses / "dwi" / f"{case_id}_{ses}_adc.nii.gz"
    mask = derivatives_case_dir / ses / f"{case_id}_{ses}_msk.nii.gz"

    for p in (dwi, adc, mask):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")
    return dwi, adc, mask


def load_volume(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    spacing = img.header.get_zooms()[:3]
    return data, spacing


def seed_everything(seed):
    """Seed everything reachable from this training run. Doesn't cover
    TorchIO's own RNG for patch sampling / augmentation — that's a
    separate seed space and left unseeded, so patch locations and
    augmentation still vary run to run even with this fixed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    """Resolve config.yaml's training.device to an actual torch.device,
    falling back automatically if the requested backend isn't available."""
    device_name = config["training"]["device"]
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_score(pred_probs, target, threshold=0.5, eps=1e-6):
    """Discrete Dice over a batch (thresholded prediction vs. binary
    target), averaged across the batch. Spatial dims are inferred as
    everything after (batch, channel), so this works for a patch batch
    (B, C, X, Y, Z) or a single full volume (1, C, X, Y, Z) alike."""
    pred = (pred_probs > threshold).float()
    dims = tuple(range(2, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred_probs, target, threshold=0.5, eps=1e-6):
    pred = (pred_probs > threshold).float()
    dims = tuple(range(2, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def sensitivity(pred_probs, target, threshold=0.5, eps=1e-6):
    """Recall / true positive rate: TP / (TP + FN). For a case with zero
    true lesion voxels (3/250 in this dataset — see dataset_stats.csv),
    this is technically undefined; the eps guard returns ~0 rather than
    NaN, which is a reasonable default but worth knowing about if such a
    case ends up in the test split and skews the mean."""
    pred = (pred_probs > threshold).float()
    dims = tuple(range(2, pred.dim()))
    tp = (pred * target).sum(dim=dims)
    fn = ((1 - pred) * target).sum(dim=dims)
    return (tp / (tp + fn + eps)).mean().item()


def precision(pred_probs, target, threshold=0.5, eps=1e-6):
    """TP / (TP + FP)."""
    pred = (pred_probs > threshold).float()
    dims = tuple(range(2, pred.dim()))
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1 - target)).sum(dim=dims)
    return (tp / (tp + fp + eps)).mean().item()
