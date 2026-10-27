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

Run the one-seed real-data GPU baseline comparison after the local ResNet
weights are cached and the cleaned Dogs vs Cats data is under `data/`. The
configuration compares ResNet-18 and ResNet-50 in head-only (`frozen`),
last-block, and full fine-tuning modes with AMP and early stopping. Run folders
and the generated sweep summary stay under ignored `runs/`; the completed
baseline sweep also writes its report table to `docs/sweep_results.md`:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/probe_vram.py --config configs/base.yaml
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/sweep.py --sweep configs/sweeps/gpu_baseline.yaml --report docs/sweep_results.md
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

Preprocess and audit the team handoff before training. Preprocessing reads
`raw-data/datasets/`, writes 224x224 JPEGs under `data/`, and removes exact
duplicates while prioritizing train over validation and test. The audit writes
`audit.json` and `audit.md` into the audited root and exits with status 1 for a
missing required directory or a train/validation duplicate:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe src/ie4483/preprocess_images.py
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

Create fixed on-disk training views with one seeded random adjustment per
training image. `augmented` contains adjusted images only; `mixed` retains each
original beside its adjusted JPEG. Validation and test files are copied
unchanged, and `augmentation.csv` records the operation and sampled parameters.
The project data outputs are ignored by git:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/make_augmented_view.py --input data --output data_aug --mode augmented --seed 0
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/make_augmented_view.py --input data --output data_mixed --mode mixed --seed 0
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
fp32. Deterministic operations are enabled by default with `train.deterministic`
in the resolved run config; set it to `false` to disable them. Frozen,
unaugmented runs reuse pooled features under `runs/feature_cache/`.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/base.yaml
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config configs/base.yaml --set train.epochs=2 --set model.pretrained=false
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/train.py --config runs/path-to-run/config.yaml --resume runs/path-to-run
```

Check repeatability by training twice with the configured seed and once with
the next seed. The command compares per-epoch loss and accuracy metrics and
exits nonzero when the same-seed difference exceeds the tolerance (default 0).
It writes three normal run folders under the selected output path.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/check_repeatable.py --config configs/base.yaml --epochs 2 --set model.pretrained=false
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/check_repeatable.py --config configs/base.yaml --epochs 2 --tolerance 0.000001 --output runs/repeatability-check
```

Training writes an atomic `last.pt` checkpoint after each completed epoch. To
continue an interrupted run, pass its folder with `--resume`; the supplied
configuration and overrides must match the saved run configuration. Seed and
sweep runners also continue an unfinished run folder when it has `last.pt`.

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

Create per-run loss and accuracy curves, an overlay of validation accuracy, and
CSV/Markdown comparison tables from completed run folders. Each run folder gets
its own `training_curves.png` and `.pdf`; all combined artifacts go to `--out`.
Proxy data runs are marked `PROXY` in the figures and table.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/report_figures.py --runs runs/run-a runs/run-b --out runs/report
```

Predict the unlabeled test split using a completed run. The command writes
`submission.csv` (numeric IDs in ascending order, `1` for dog) and
`test_predictions.csv` into the run folder. Optional horizontal-flip TTA averages
the original and flipped image probabilities.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/predict.py --run runs/path-to-run
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/predict.py --run runs/path-to-run --tta flip
```

Regenerate the team report's text summary from the finished sweep and seed runs,
then execute `notebooks/cats_vs_dogs_report.ipynb` from the repository root. The
notebook reads only `runs/report/summary/` and optional local images from `data/`;
its HTML share copy is written under ignored `runs/report/`.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/export_report_summary.py
jupyter nbconvert --execute --to html --output-dir runs/report notebooks/cats_vs_dogs_report.ipynb
```

Reproduce the selected ResNet-50 frozen configuration across seeds 0, 1, and 2
after the owner has prepared the real data and local weights. The result report
and copied figures are in `docs/results.md` and `docs/figures/`; the root
`submission.csv` was generated from the seed with the highest validation accuracy.
Use the saved seed run folders with the evaluation, prediction, and figure-report
commands above. The `runs/` artifacts are local and are not committed.

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/run_seeds.py --config configs/base.yaml --seeds 0 1 2 --set model.architecture=resnet50 --set model.freeze_mode=frozen --set train.batch_size=71 --set train.gradient_accumulation_steps=1 --set train.mixed_precision=true --set train.epochs=10 --set train.patience=3 --output runs/seeds/best
```

## Tests

Run the CPU test suite with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q
```
