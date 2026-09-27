"""
Day 3-4 — training loop.

Owns:
    - Wiring dataset.py's patch loaders to model.py's UNet3D.
    - Combined Dice + BCE loss (soft/differentiable Dice for the loss;
      utils.py's discrete dice_score is used separately for the val
      metric that decides checkpointing).
    - Mixed precision: verified once at startup with a dummy forward pass,
      but that alone isn't sufficient — MPS autocast produced NaN loss
      after enough real training steps in practice, so config.yaml's
      training.mixed_precision now defaults to false. Gradient clipping
      (max_norm=1.0) and a fail-fast check on non-finite loss are both
      in place as further insurance, whichever precision mode is used.
    - Checkpointing: checkpoints/best.pt (best val Dice so far) and
      checkpoints/last.pt (every epoch, for resuming after a crash).
    - Early stopping on training.early_stopping_patience (val Dice not
      improving — a model-quality stop), plus an independent optional cap
      on training.max_steps (a total-step compute budget — a coarser
      safety net that stops the run regardless of whether Dice is still
      improving, checked both between epochs and mid-epoch).
    - THE FIRST-EPOCH TIMING PRINT: this is the real-world version of Day
      1's proxy-model estimate — logged loudly, first thing, since it's
      the number that actually confirms (or corrects) that Day 1 call.

Usage:
    python train.py

Outputs:
    - checkpoints/best.pt, checkpoints/last.pt
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import StrokeLesionPatchDataset
from model import UNet3D
from utils import load_config, seed_everything, get_device, dice_score


def dice_loss(logits, target, eps=1e-6):
    """Soft Dice — differentiable, unlike the thresholded dice_score in
    utils.py used for the reported validation metric."""
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def dice_bce_loss(logits, target, bce_weight=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    d_loss = dice_loss(logits, target)
    return bce_weight * bce + (1 - bce_weight) * d_loss


def check_mixed_precision(model, device, in_channels, patch_size):
    """One dummy forward pass under autocast, before committing the whole
    run to it. Returns False (and explains why) rather than letting a
    silent dtype issue eat an epoch before surfacing."""
    try:
        dummy = torch.randn(1, in_channels, *patch_size, device=device)
        with torch.autocast(device_type=device.type, enabled=True):
            _ = model(dummy)
        print(f"Mixed precision autocast verified working on {device.type}")
        return True
    except Exception as e:
        print(f"Mixed precision autocast failed on {device.type} ({e})")
        print("Falling back to full precision for this run.")
        return False


def run_epoch(model, loader, optimizer, device, mixed_precision, train_mode, log_every=10, label="", max_steps=None):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    epoch_start = time.time()

    for image, mask in loader:
        if max_steps is not None and n_batches >= max_steps:
            print(f"    [{label}] reached step budget ({max_steps}) — stopping this epoch early")
            break

        image, mask = image.to(device), mask.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                logits = model(image)
                loss = dice_bce_loss(logits, mask)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({loss.item()}) at {label} step {n_batches + 1}. "
                f"Stopping immediately rather than continuing on corrupted weights — "
                f"restart training after addressing the cause (see train.py's module "
                f"docstring / config.yaml's training.mixed_precision comment)."
            )

        with torch.no_grad():
            probs = torch.sigmoid(logits.float())
            batch_dice = dice_score(probs, mask)

        total_loss += loss.item()
        total_dice += batch_dice
        n_batches += 1

        if n_batches % log_every == 0:
            elapsed = time.time() - epoch_start
            print(
                f"    [{label}] step {n_batches:4d} | "
                f"running loss {total_loss / n_batches:.4f} dice {total_dice / n_batches:.4f} | "
                f"{elapsed:.1f}s elapsed ({elapsed / n_batches:.2f}s/step)"
            )

    return total_loss / n_batches, total_dice / n_batches, n_batches


def save_checkpoint(path, epoch, model, optimizer, val_dice, config=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_dice": val_dice,
    }
    if config is not None:
        payload["config"] = config
    torch.save(payload, path)


def main():
    cfg = load_config()
    seed_everything(cfg["data"]["seed"])
    device = get_device(cfg)
    print(f"Training on device: {device}")

    print("\nBuilding train/val patch loaders...")
    train_ds = StrokeLesionPatchDataset(split="train", config=cfg)
    val_ds = StrokeLesionPatchDataset(split="val", config=cfg)
    # Note: val Dice below is measured on randomly-sampled patches, same as
    # training — not a full-volume metric. It's a useful signal for
    # checkpointing/early stopping, but the number that actually matters
    # for the write-up is Day 5's full-volume evaluation on the held-out
    # test set (evaluate.py), which this is not a substitute for.

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        depth=cfg["model"]["depth"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    mixed_precision = cfg["training"]["mixed_precision"] and device.type in ("cuda", "mps")
    if mixed_precision:
        mixed_precision = check_mixed_precision(
            model, device, cfg["model"]["in_channels"], cfg["patching"]["patch_size"]
        )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    epochs_without_improvement = 0
    patience = cfg["training"]["early_stopping_patience"]
    n_epochs = cfg["training"]["epochs"]
    max_steps = cfg["training"].get("max_steps")  # None = no cap; else total TRAINING steps across all epochs
    total_train_steps = 0

    budget_str = f"{max_steps} training steps" if max_steps is not None else "no step cap"
    print(f"\nStarting training: {n_epochs} epochs (early stop patience {patience}, {budget_str})\n")

    for epoch in range(1, n_epochs + 1):
        if max_steps is not None and total_train_steps >= max_steps:
            print(f"\nReached max_steps budget ({max_steps} total training steps); stopping.")
            break

        print(f"\nEpoch {epoch}/{n_epochs} starting... (train steps so far: {total_train_steps})")
        epoch_start = time.time()

        remaining_steps = None if max_steps is None else max_steps - total_train_steps
        train_loss, train_dice, train_steps = run_epoch(
            model, train_loader, optimizer, device, mixed_precision,
            train_mode=True, label="train", max_steps=remaining_steps,
        )
        total_train_steps += train_steps

        val_loss, val_dice, _ = run_epoch(
            model, val_loader, optimizer, device, mixed_precision, train_mode=False, label="val"
        )

        epoch_time = time.time() - epoch_start

        if epoch == 1:
            est_total_min = epoch_time * n_epochs / 60
            print(
                f">>> First epoch took {epoch_time:.1f}s — "
                f"est. total for {n_epochs} epochs: {est_total_min:.1f} min\n"
            )

        print(
            f"Epoch {epoch:3d} | train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} | {epoch_time:.1f}s | "
            f"total train steps {total_train_steps}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, val_dice, cfg)
            print(f"  -> new best val dice {val_dice:.4f}, saved to {checkpoint_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        save_checkpoint(checkpoint_dir / "last.pt", epoch, model, optimizer, val_dice)

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no val Dice improvement for {patience} epochs.")
            break

    print(f"\nTraining done. Best val Dice: {best_val_dice:.4f} ({checkpoint_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
