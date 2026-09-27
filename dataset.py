"""
Day 2 — PyTorch Dataset + patch sampling.

Owns:
    - Loading each preprocessed case (data/processed/<case_id>/{dwi,adc,mask}.nii.gz)
      into a TorchIO Subject: DWI+ADC stacked into one 2-channel image, mask
      as a label map.
    - Foreground-biased patch sampling via TorchIO's LabelSampler, honoring
      config.yaml's `patching.foreground_ratio` — this is what prevents the
      model from collapsing to all-background predictions given how small
      lesions are relative to full brain volume.
    - Data augmentation (flips, affine, noise) for the train split only;
      val/test get no augmentation. (RandomElasticDeformation was tried
      and dropped — see get_augmentation_transform()'s docstring.)
    - A runnable sanity check (`python dataset.py`) that pulls one batch and
      saves a mid-slice overlay figure, so patch alignment and foreground
      bias can be eyeballed before Day 3's training loop depends on it.

    2.5D mode (config.yaml's `patching.mode: "2p5d"`) is NOT implemented —
    Day 1's compute check came back "full 3D feasible", so the fallback
    path was never needed. If that changes, this file needs a second
    __getitem__ path here before train.py can use it.

Usage:
    from dataset import StrokeLesionPatchDataset
    train_ds = StrokeLesionPatchDataset(split="train")
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torchio as tio

from utils import load_config


def load_case_subject(case_dir, min_size=None):
    """Build a tio.Subject for one preprocessed case: a 2-channel image
    (DWI, ADC stacked) plus the lesion mask as a label map. DWI/ADC/mask
    share the same affine — preprocess.py's save_case() gives all three
    the same target_spacing-derived affine, so no re-alignment is needed
    here.

    If min_size is given, pads any spatial dimension smaller than it up to
    exactly min_size (never crops — target is elementwise max of current
    shape and min_size). Brain size varies by case, so after resampling
    some volumes end up smaller than the patch size along one axis; this
    is what TorchIO's sampler needs to not immediately fail on those."""
    case_dir = Path(case_dir)

    dwi_img = nib.load(str(case_dir / "dwi.nii.gz"))
    adc_img = nib.load(str(case_dir / "adc.nii.gz"))
    mask_img = nib.load(str(case_dir / "mask.nii.gz"))

    dwi = dwi_img.get_fdata(dtype=np.float32)
    adc = adc_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32)

    image_tensor = torch.from_numpy(np.stack([dwi, adc], axis=0))       # (2, X, Y, Z)
    mask_tensor = torch.from_numpy(mask[np.newaxis, ...])               # (1, X, Y, Z)

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image_tensor, affine=dwi_img.affine),
        mask=tio.LabelMap(tensor=mask_tensor, affine=dwi_img.affine),
        case_id=case_dir.name,
    )

    if min_size is not None:
        current_shape = subject.spatial_shape
        target_shape = tuple(max(c, m) for c, m in zip(current_shape, min_size))
        if target_shape != current_shape:
            subject = tio.CropOrPad(target_shape)(subject)

    return subject


def build_subjects(processed_dir, case_ids, min_size=None):
    subjects = []
    skipped = []
    for case_id in case_ids:
        case_dir = Path(processed_dir) / case_id
        try:
            subjects.append(load_case_subject(case_dir, min_size=min_size))
        except Exception as e:
            skipped.append((case_id, str(e)))
    if skipped:
        print(f"  {len(skipped)} case(s) skipped loading subjects:")
        for case_id, err in skipped[:5]:
            print(f"    - {case_id}: {err}")
    return subjects


def get_augmentation_transform():
    """Augmentation for the training split only. RandomElasticDeformation
    was tried and dropped: it warps the *full volume* on every subject
    load (TorchIO applies subject-level transforms before patch
    extraction), and with 175 training subjects that made the Queue's
    initial buffer fill take upward of 12 hours. Flip/affine/noise cover
    most of the overfitting-reduction benefit at a small fraction of the
    cost."""
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.3),
        tio.RandomNoise(std=(0, 0.05), p=0.2),
    ])


class StrokeLesionPatchDataset(torch.utils.data.Dataset):
    """Wraps a TorchIO Queue of foreground-biased patches for one split as
    a standard (map-style) Dataset — this is TorchIO's own documented
    usage pattern: Queue already implements __len__/__getitem__, sized to
    len(subjects) * patches_per_volume patches per epoch, and refills
    internally regardless of the index requested.

    (An earlier version of this file made it an IterableDataset instead,
    iterating the Queue directly with `for patch in self.queue: ...`.
    That loop doesn't respect the Queue's per-epoch length — it just kept
    pulling patches indefinitely, which is why training blew past step
    4000 in what should have been a ~700-step epoch. __getitem__ is the
    correct interface here, not __iter__.)
    """

    def __init__(self, split, config=None):
        assert split in ("train", "val", "test")
        self.config = config or load_config()
        self.split = split

        processed_dir = self.config["data"]["processed_dir"]
        with open(Path(processed_dir) / "splits.json") as f:
            case_ids = json.load(f)[split]

        patch_size = tuple(self.config["patching"]["patch_size"])

        subjects = build_subjects(processed_dir, case_ids, min_size=patch_size)
        if not subjects:
            raise RuntimeError(
                f"No subjects loaded for split '{split}'. Did you run preprocess.py?"
            )

        transform = get_augmentation_transform() if split == "train" else None
        subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)

        foreground_ratio = self.config["patching"]["foreground_ratio"]
        sampler = tio.LabelSampler(
            patch_size=patch_size,
            label_name="mask",
            label_probabilities={0: 1 - foreground_ratio, 1: foreground_ratio},
        )

        patches_per_volume = self.config["patching"]["patches_per_volume"]
        # max_length is a prefetch BUFFER, not meant to scale with dataset
        # size — it was previously patches_per_volume * len(subjects) * 4
        # (5600 for the train split), which forced ~700 full-volume
        # transform passes before the first patch was ever yielded. Capped
        # here at a fixed, modest size regardless of subject count.
        self.queue = tio.Queue(
            subjects_dataset=subjects_dataset,
            max_length=min(300, patches_per_volume * len(subjects)),
            samples_per_volume=patches_per_volume,
            sampler=sampler,
            shuffle_subjects=(split == "train"),
            shuffle_patches=(split == "train"),
        )

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        patch = self.queue[index]
        image = patch["image"].data.float()   # (2, pX, pY, pZ)
        mask = patch["mask"].data.float()      # (1, pX, pY, pZ)
        return image, mask


# ---------------------------------------------------------------------------
# Sanity check — run directly: python dataset.py
# ---------------------------------------------------------------------------

def _sanity_check():
    import matplotlib.pyplot as plt

    cfg = load_config()
    ds = StrokeLesionPatchDataset(split="train", config=cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    image, mask = next(iter(loader))
    print(f"image batch shape: {tuple(image.shape)}  (batch, channels, X, Y, Z)")
    print(f"mask batch shape:  {tuple(mask.shape)}")
    print(f"foreground voxels in this patch: {int(mask.sum())} "
          f"(0 is possible — foreground_ratio is probabilistic per patch, not guaranteed)")

    mid = image.shape[-1] // 2
    dwi_slice = image[0, 0, :, :, mid].numpy()
    mask_slice = mask[0, 0, :, :, mid].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(dwi_slice, cmap="gray")
    axes[0].set_title("DWI patch (mid slice)")
    axes[1].imshow(dwi_slice, cmap="gray")
    axes[1].imshow(mask_slice, cmap="Reds", alpha=0.4)
    axes[1].set_title("with mask overlay")
    for ax in axes:
        ax.axis("off")

    out_path = Path("results/figures/patch_sanity_check.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved sanity check figure to {out_path}")


if __name__ == "__main__":
    _sanity_check()
