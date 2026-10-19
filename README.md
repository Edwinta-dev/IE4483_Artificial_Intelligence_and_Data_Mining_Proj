# IE4483 Artificial Intelligence and Data Mining Project

Dogs vs Cats image classification by transfer learning from ResNet-18/50 with a
custom classifier head, followed by CIFAR-10 and an imbalanced CIFAR-10 study.
This repository is building the training and evaluation pipeline; the initial
setup does not train on real data or download weights.

## Environment

Use the shared environment at `C:/Users/AgenticDev/ml-venv/Scripts/python.exe`.
The data root is `data/` by default, can be set with `IE4483_DATA`, or can be
configured in `configs/base.yaml`. Data, run outputs, and model weights are
ignored by git.

## Setup

Put the provided Dogs vs Cats dataset under `data/` using `train/cat`,
`train/dog`, `val/cat`, `val/dog`, and `test/` folders. Put the team's cleaned
training images in the corresponding `train/cat` and `train/dog` folders. To
store the dataset elsewhere, set `IE4483_DATA` or set `data.root` in the YAML
config. The config value takes precedence over the environment variable.

The owner can cache torchvision's ResNet-18 and ResNet-50 ImageNet weights
once, and check the local setup, with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/fetch_weights.py
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/check_setup.py
```

`fetch_weights.py` is the only setup command that downloads files. It respects
`TORCH_HOME`; `check_setup.py` reports the data layout, cached weight files,
Python/PyTorch versions, and whether CUDA can allocate a small tensor. To check
a specific root, pass `--data-root PATH`; to resolve it from a config, pass
`--config PATH`.

Audit a cleaned data handoff before training. The command writes `audit.json`
and `audit.md` into the audited root and exits with status 1 for a missing
required directory or a train/validation duplicate:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/audit_data.py --root data
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/audit_data.py --root data --compare raw-data
```

Preview the configured training augmentations for a single image. The contact
sheet is written under `runs/` by default; `--output` can point into a specific
run folder. Edit the `augment` switches in `configs/base.yaml` to compare
pipelines, or leave them all disabled to use the evaluation preprocessing.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/preview_augment.py --image path/to/image.jpg --n 8 --output runs/augmentation_preview.png
```

Train a configured ResNet transfer-learning run. Each run writes its resolved
configuration, transform description, file manifest, epoch metrics, and best
checkpoint under `runs/`. The default batch size is 32 for a frozen ResNet-50
at 224 pixels on 4 GB VRAM; use batch size 4 when fine tuning the backbone and
increase gradient accumulation if needed. CUDA uses AMP when enabled; CPU uses
fp32. Frozen, unaugmented runs reuse pooled features under `runs/feature_cache/`.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/base.yaml
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/base.yaml --set train.epochs=2 --set model.pretrained=false
```

## Tests

Run the CPU test suite with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q
```
