# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PITA (Photo-z Inference with Triple-loss Algorithm) is a semi-supervised, image-based photometric redshift (photo-z) algorithm for galaxies. It uses MoCo-style contrastive learning jointly with redshift prediction and photometric color prediction losses to train on both labeled and unlabeled galaxy images. Cal-PITA extends PITA to output calibrated redshift probability distributions (CDEs) via the [Cal-PIT](https://github.com/lee-group-cmu/Cal-PIT) framework.

## Installation

```bash
pip install .
# or
pip install pita-z
```

## Running Training Scripts

All training scripts take a config name (without `.yaml`) and a run number:

```bash
python scripts/pita_training.py pita_default 1
python scripts/fully_supervised_training.py fully_supervised_default 1
python scripts/calpita_training.py <config_name> <run_number>
python scripts/calpit_fully_supervised_training.py <config_name> <run_number>
python scripts/calpit_photometry_training.py <config_name> <run_number>
```

Scripts read config from `configs/<config_name>.yaml`. By default, training runs on 4 GPUs using PyTorch Lightning DDP with mixed precision.

## Architecture

### Models (`src/pita_z/models/`)

**`pita_model.py`** — Two main Lightning modules:
- `PITALightning`: Semi-supervised MoCo model. Architecture: CNN encoder → optional `encoder_mlp` (projects to latent space) → `projection_head` (contrastive), `redshift_mlp` (point estimate), `color_mlp` (photometric colors). Total loss = CL loss + Huber redshift loss + MSE color loss. A frozen momentum encoder (EMA) maintains a queue of negative samples.
- `CalPITALightning`: Cal-PITA extension. The `redshift_mlp` receives a random alpha (uniform in [0,1]) concatenated to the latent vector and outputs a PIT value. Loss is BCE on the binary variable W = (true_PIT ≤ alpha). CDEs are derived by converting the learned CDF via `PchipInterpolator`.

**`fully_supervised_model.py`** — Three Lightning modules:
- `CNNPhotoz`: Fully supervised CNN for point-estimate photo-z (Huber loss).
- `CalpitCNNPhotoz`: Fully supervised Cal-PIT CNN for CDE outputs.
- `CalpitPhotometryLightning`: Cal-PIT model operating on photometric features (not images).

**`basic_models.py`** — Building blocks:
- `MLP`: Standard MLP with ReLU activations.
- `LipschitzMLP` / `MonotonicMLP`: Lipschitz-bounded MLP (using `monotonicnetworks`), wrapped for monotonicity; used as `redshift_mlp` in Cal-PIT models to ensure monotonic CDFs.
- `Encoder`, `ConvBlock`, `JointBlocks`: Custom CNN encoder.
- `CustomConvNeXt`: Wrapper for `torchvision.models.convnext_tiny` with custom first-layer input channels.

The default encoder in all scripts is `convnext_tiny` with the first conv layer replaced to accept `n_filters` input channels (typically 4 photometric bands).

### Data (`src/pita_z/data_modules/data_modules.py`)

Datasets read from HDF5 files. Expected HDF5 keys:
- `images`: galaxy image cutouts `(C, H, W)`
- `redshifts`: spectroscopic redshifts
- `ebvs`: Milky Way E(B-V) extinction values
- `dered_color_features`: dereddened photometric colors/magnitudes
- `use_redshift_<label_f>`: binary weight (1 = has redshift label, 0 = unlabeled)

Key classes:
- `ImagesDataset` / `ImagesDataModule`: For PITA and fully supervised CNN models.
- `CalpitImagesDataset` / `CalpitImagesDataModule`: For Cal-PITA; requires pre-computed PIT values.
- `CalpitPhotometryDataset` / `CalpitPhotometryDataModule`: For photometry-only Cal-PIT.

`label_f` controls the labeled fraction: `1/label_f` of data is labeled. The dataset loads `use_redshift_{label_f}` as the weight array.

### Utilities (`src/pita_z/utils/`)

- `reddening.py` — `ReddeningTransform`: Corrects galaxy images for MW dust extinction using per-band R values and E(B-V). Applied at dataset load time (not as a training augmentation by default).
- `augmentations.py` — `JitterCrop`: Randomly jitter-crops image around center. `AddGaussianNoise`: Adds per-band Gaussian noise (std = per-band MAD).
- `lr_schedulers.py` — `WarmupCosine`: Linear warmup then cosine decay. `WarmupCosineAnnealingScheduler`: Linear warmup then cosine annealing that plateaus at `min_lr`.

### Config Files (`configs/`)

YAML files control all hyperparameters. Key sections:
- `data`: HDF5 paths, batch size, num workers, `n_filters`, `label_f`
- `augmentations`: R values for extinction correction, crop dim, jitter, Gaussian noise flag
- `model`: encoder name, `latent_d`, `projection_d`, MLP hidden layers, queue size, temperature
- `training`: epochs, learning rate, momentum, loss weights, `lr_scheduler` (type + per-scheduler params)
- `logging_and_checkpoint`: TensorBoard log dir, checkpoint dir, checkpoint frequency

Available lr_scheduler types: `cosine`, `multistep`, `warmupcosine`, `wc_ann`, `None`.

### Logged Metrics

All models log: `training_bias`, `training_nmad`, `training_outlier_f` (photo-z metrics using Δz/(1+z)). PITA/CalPITA also log contrastive loss components (`cl_training_loss`, `training_pos_sim`).

## Package Structure

```
src/pita_z/
├── __init__.py
├── models/
│   ├── basic_models.py       # MLP, encoder, building blocks
│   ├── pita_model.py         # PITALightning, CalPITALightning
│   └── fully_supervised_model.py  # CNNPhotoz, CalpitCNNPhotoz, CalpitPhotometryLightning
├── data_modules/
│   └── data_modules.py       # Dataset and DataModule classes
└── utils/
    ├── augmentations.py      # JitterCrop, AddGaussianNoise
    ├── reddening.py          # ReddeningTransform
    └── lr_schedulers.py      # WarmupCosine, WarmupCosineAnnealingScheduler
scripts/
├── pita_training.py
├── fully_supervised_training.py
├── calpita_training.py
├── calpit_fully_supervised_training.py
└── calpit_photometry_training.py
configs/
├── pita_default.yaml
└── fully_supervised_default.yaml
```
