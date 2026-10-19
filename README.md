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

For the CIFAR-10 extension, the owner can fetch the standard Python batches once
with `scripts/fetch_cifar10.py`. `configs/cifar10.yaml` uses the ten named
classes, a 96-pixel input, and CIFAR channel statistics instead of the binary
cat/dog head and ImageNet normalization. The 32-pixel images are enlarged for
ResNet-18; training adds a four-pixel padded random crop and horizontal flip.
The fixed seed carves validation samples from the training batches. Validation
selects the model; the official labeled test batch is reserved for final
evaluation:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/fetch_cifar10.py
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/cifar10.yaml
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/evaluate.py --run runs/path-to-run --split test
```

Evaluation writes per-class scores and a ten-by-ten confusion matrix in the
run folder, including macro-F1 and macro recall. No dataset is downloaded by tests.

To exercise the binary Dogs vs Cats pipeline before the cleaned images arrive,
the owner can fetch CIFAR-10 and materialize its cat (class 3) and dog (class 5)
images as a marked stand-in dataset. The generator uses the existing seeded
training-batch split, writes an unlabeled numeric-ID test folder from
`test_batch`, and records source identities and hashes in a manifest. Its
metrics include `"proxy": true`; these scores are only a pipeline check and
must never be reported as Dogs vs Cats results. This owner workflow needs
network access for the one-time CIFAR fetch and a GPU for practical 224-pixel
training, although the generator and pipeline tests run on CPU.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/fetch_cifar10.py
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/fetch_weights.py
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/make_proxy_catdog.py --cifar-root data --out data/proxy-catdog
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/proxy_catdog.yaml --set data.root=data/proxy-catdog
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/evaluate.py --run runs/path-to-proxy-run
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/predict.py --run runs/path-to-proxy-run
```

For the CIFAR-10 class-imbalance study, `configs/sweeps/imbalance.yaml` compares
balanced and class-1-at-10% training subsets across seeds and remedy switches.
Use `train.imbalance_fractions` as a label-to-fraction mapping (unlisted labels
stay complete), `train.class_weighted` for normalized inverse-frequency
cross-entropy weights, `train.weighted_sampler` for balanced sampling, and
`train.focal_gamma` to enable focal loss. These switches can be combined. Each
run records its selected training counts in `class_counts.json`; validation and
test sets are unchanged. Run it with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/sweep.py --sweep configs/sweeps/imbalance.yaml
```

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

Probe CUDA memory before training to find a safety-adjusted batch for each
ResNet-18/50 freeze mode. The command uses synthetic batches and local random
weights, writes `vram_probe.json` under `runs/`, and does not need project data.
It uses seed 42 by default; pass `--seed` to reproduce a different probe.
Pass `--config` to use its configured image size; repeat it to update several
configs with the measured batch for each config's backbone and freeze mode.
`--write-configs` saves those batch and gradient accumulation values in place.
The probe reports a message and exits when CUDA is unavailable.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/probe_vram.py --config configs/base.yaml
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/probe_vram.py --config configs/base.yaml --config configs/proxy_catdog.yaml --write-configs
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

Evaluate a completed run on its labeled validation split. This writes metrics,
predictions, a confusion matrix, per-class confidence galleries, and
`results/latest.json` (override its location with `--results`).

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/evaluate.py --run runs/path-to-run
```

Repeat a training and validation run across fixed seeds. The command writes
per-seed results plus `summary.csv` and `summary.md` under the output folder.
Use `--set KEY=VALUE` to apply normal training config overrides.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/run_seeds.py --config configs/base.yaml --seeds 0 1 2
```

Run a parameter grid from `configs/sweeps/`. Each parameter combination gets
the listed seeds, and completed seed runs with `val_metrics.json` are reused.
The output contains sorted summary tables and `sweep.png` with standard
deviation error bars. Optional shared `overrides: [train.batch_size=4]` entries
in the sweep YAML apply fixed config values to every cell. `--output` selects a
different artifact folder.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/sweep.py --sweep configs/sweeps/learning_rate.yaml
```

Create the architecture diagram (PNG and PDF) and model card from a config.
Set `--output` to the run folder where these report artifacts should live.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/model_figure.py --config configs/base.yaml --output runs/report
```

Predict the unlabeled test split using a completed run. The command writes
`submission.csv` (numeric IDs in ascending order, `1` for dog) and
`test_predictions.csv` into the run folder. Optional horizontal-flip TTA averages
the original and flipped image probabilities.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/predict.py --run runs/path-to-run
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/predict.py --run runs/path-to-run --tta flip
```

## Tests

Run the CPU test suite with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q
```
