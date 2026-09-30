# Ischemic Stroke Lesion Segmentation — 3D U-Net (ISLES 2022)

Patch-based 3D U-Net for ischemic stroke lesion segmentation on
DWI/ADC brain MRI, benchmarked against published ISLES'22 baselines.


## Project structure

```
stroke_lesion_unet/
├── README.md
├── requirements.txt
├── config.yaml               # all tunables in one place: paths, patch size, model, training
├── download_data.sh          # fetches ISLES-2022.zip from Zenodo, verifies checksum
├── explore.py                # dataset stats — dims, spacing, intensity, lesion volume dist.
├── preprocess.py             #  crop to brain bbox, normalize, resample, train/val/test split
├── dataset.py                # PyTorch Dataset + foreground-biased patch sampler (TorchIO)
├── model.py                  # reduced 3D U-Net (also holds the 2.5D fallback model)
├── train.py                  # training loop, checkpointing, Dice+BCE loss
├── evaluate.py                # sliding-window inference, Dice/IoU vs. baselines
├── visualize.py               # overlay figures, results table
├── utils.py                   # shared helpers (metrics, I/O, seeding)
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

- **Dataset:** ISLES 2022 training set, 250 cases (DWI + ADC used; FLAIR dropped)
- **Source:** https://zenodo.org/records/7153326 (1.7 GB, CC BY 4.0)
- **Citation:** Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U. et al.
  "ISLES 2022: A multi-center magnetic resonance imaging stroke lesion
  segmentation dataset." Sci Data 9, 762 (2022).
- **Split:** seeded 70/15/15 → 175 train / 38 val / 37 test (`data/processed/splits.json`)
- **Spacing:** highly non-uniform across cases (7 distinct spacing profiles,
  slice thickness 2.0–4.8mm) — resampling to the dataset's median spacing,
  `[2.0, 2.0, 2.0]` mm, before patch extraction
