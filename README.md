# Ischemic Stroke Lesion Segmentation — 3D U-Net (ISLES 2022)

Portfolio project: patch-based 3D U-Net for ischemic stroke lesion segmentation on
DWI/ADC brain MRI, benchmarked against published ISLES'22 baselines.

> Results, figures, and the full write-up will be filled in as the week progresses.
> This README is the living index — update it each day rather than writing a
> separate log.

## Status

- [x] Day 1 — data downloaded, environment set up, exploration done, compute go/no-go timed
- [ ] Day 2 — preprocessing + patch sampling pipeline
- [ ] Day 3 — model implemented, training started, feasibility checkpoint
- [ ] Day 4 — training continued / fallback executed if needed
- [ ] Day 5 — final model, full-volume evaluation vs. baselines
- [ ] Day 6 — figures and visual deliverables
- [ ] Day 7 — write-up, repo polish, ship

## Project structure

Flat layout — everything at the top level, no `src/`/`configs/` nesting:

```
stroke_lesion_unet/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml               # all tunables in one place: paths, patch size, model, training
├── download_data.sh          # fetches ISLES-2022.zip from Zenodo, verifies checksum
├── explore.py                # Day 1: dataset stats — dims, spacing, intensity, lesion volume dist.
├── preprocess.py             # Day 2: crop to brain bbox, normalize, resample, train/val/test split
├── dataset.py                # Day 2: PyTorch Dataset + foreground-biased patch sampler (TorchIO)
├── model.py                  # Day 3: reduced 3D U-Net (also holds the 2.5D fallback model)
├── train.py                  # Day 3-4: training loop, checkpointing, Dice+BCE loss
├── evaluate.py                # Day 5: sliding-window inference, Dice/IoU vs. baselines
├── visualize.py               # Day 6: overlay figures, results table
├── utils.py                   # shared helpers (metrics, I/O, seeding)
├── data/
│   ├── raw/                  # ISLES-2022 extracted here (gitignored, not committed)
│   └── processed/            # splits.json + cropped/normalized volumes (gitignored)
├── checkpoints/               # saved model weights (gitignored)
└── results/
    ├── figures/               # final overlay images, comparison plots for README/site
    └── metrics/                # dice_scores.csv, results_table.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash download_data.sh
python explore.py
python preprocess.py
```

## Data

- **Dataset:** ISLES 2022 training set, 250 cases (DWI + ADC used; FLAIR dropped —
  see limitations in the write-up)
- **Source:** https://zenodo.org/records/7153326 (1.7 GB, CC BY 4.0)
- **Citation:** Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U. et al.
  "ISLES 2022: A multi-center magnetic resonance imaging stroke lesion
  segmentation dataset." Sci Data 9, 762 (2022).
- **Split:** seeded 70/15/15 → 175 train / 38 val / 37 test (`data/processed/splits.json`)
- **Spacing:** highly non-uniform across cases (7 distinct spacing profiles,
  slice thickness 2.0–4.8mm) — resampling to the dataset's median spacing,
  `[2.0, 2.0, 2.0]` mm, before patch extraction
- **Compute check:** MPS timing test on the target patch size (64³) estimates
  ~0.1 min/epoch — full 3D patch training is feasible, no fallback to 2.5D needed

## Results

_Filled in on Day 5-6._

| Model | Dice | IoU | Notes |
|---|---|---|---|
| This project | — | — | — |
| ISLES'22 top team (SegResNet ensemble) | 0.824 | — | published, 15-model ensemble |

## Limitations

_Filled in on Day 7 — be specific: modalities used, split strategy, compute
constraints, single vs. cross-validated run._
