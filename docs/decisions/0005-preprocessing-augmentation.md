# 0005: Preprocessing and augmentation preview
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #5 · Date: 2026-10-19
## In short
The package now provides ImageNet-normalized evaluation preprocessing and a configurable training pipeline with crop, flip, rotation, and color-jitter switches. A command produces a de-normalized contact sheet from one input image, and the transform description can be stored with run metadata by callers.
## How it works
1. The caller loads the YAML settings, including the image size and augmentation choices. Each augmentation has its own switch; its related values control the crop range and color changes.
2. For evaluation, the image is resized so its shorter edge reaches the configured size, center-cropped to a square, converted to a float tensor, and normalized with ImageNet channel statistics.
3. For training, enabled augmentation steps are placed before tensor conversion and normalization. If all switches are off, the same deterministic resize and center crop used by evaluation are applied.
4. The description helper returns both pipelines and their settings as readable text. A caller can write that text beside the run configuration for a report or later comparison.
5. The preview command loads one image, applies the training pipeline repeatedly, reverses normalization for display, and saves the views as a PNG grid. Its output defaults under `runs/`, and its configuration and destination can be selected from the command line.
## Main design choice
The transforms use torchvision's standard image operations, which directly express the requested order and parameters and return PyTorch tensors expected by a classifier. A custom PIL/torch implementation could reduce reliance on torchvision transform behavior, but would duplicate well-tested image operations and make future transform comparisons harder to read. The switches live in a typed `augment` config section; extending the existing strict loader was necessary because it rejects unknown keys. Keeping settings as untyped YAML in the new module would avoid that small schema change, but would bypass validation and config round-tripping used elsewhere in the project.
## Traps: why the obvious fix is wrong
- Normalization must happen after augmentation and tensor conversion. Applying geometric or color operations to normalized tensors changes their meaning and can make PIL operations unusable.
- The preview reverses channel normalization before plotting; otherwise the report image has clipped, misleading colors.
- `Resize(size)` in torchvision targets the shorter edge, so evaluation needs a separate center crop to produce the model's square input.
- Transform construction calls the shared seed helper with the configured training seed (or an explicit seed). Random operations still vary on repeated calls; resetting the global seed on every image would make a batch repeat one identical augmentation.
- An empty augmentation configuration follows the evaluation path, making the no-augmentation comparison deterministic.
## Components
- `src/ie4483/transforms.py::build_eval_transform` — eval pipeline
- `src/ie4483/transforms.py::build_train_transform` — seeded training pipeline
- `src/ie4483/transforms.py::describe_transforms` — readable listing
- `src/ie4483/config.py::AugmentConfig` — typed controls
- `scripts/preview_augment.py::main` — PNG grid command
- `configs/base.yaml` — defaults
- `tests/test_transforms.py` — output, switches, preview
## Breaks or needs follow-up
None known. The strict configuration schema has a new `augment` section; existing configs that omit it retain disabled defaults.
## Problem and constraints
This pipeline groundwork must work on CPU, avoid real data and pretrained weights, and make augmentation easy to describe and switch for later controlled comparisons. The preview operates on a caller-provided local image and writes a report artifact, not a dataset or model output.
## Alternatives considered
- One all-or-nothing augmentation mode — smaller config, but cannot compare individual operations or parameter settings.
- Use evaluation preprocessing directly for `augment: none` — this behavior is retained as the no-enabled-switch path, while a single shared resize/crop implementation keeps train and evaluation geometry aligned.
- Hand-written matplotlib/PIL transforms — no benefit over torchvision here and risk of different image semantics.
## Trade-offs
Seeding at transform construction also initializes the shared Python, NumPy, and torch streams, consistent with project reproducibility conventions. Applications constructing transforms mid-run should account for this global RNG state change. Preview creation depends on matplotlib, which is present in the project environment but is not a runtime dependency of the training transform module.
## Key parameters
- `image_size: 224` (`configs/base.yaml::data.image_size`) sets the final square crop; larger values retain more detail and increase model compute.
- `random_resized_crop_scale: [0.8, 1.0]` (`configs/base.yaml::augment.random_resized_crop_scale`) controls sampled crop area; lower bounds further below 1 create stronger zoom/crop variation.
- `horizontal_flip: false` (`configs/base.yaml::augment.horizontal_flip`) enables a 0.5 probability flip when true.
- `rotation_degrees: 0.0` (`configs/base.yaml::augment.rotation_degrees`) is the symmetric rotation limit; greater values permit stronger rotations.
- Color jitter switch and strengths (`configs/base.yaml::augment.color_jitter*`) default to disabled and brightness/contrast/saturation 0.2, hue 0.0; higher strengths produce more color variation.
- `--n 8` (`scripts/preview_augment.py::main`) controls the number of displayed samples; larger grids create larger report figures.
## Assumptions
Inputs are RGB-readable images, matching the existing dataset loader's output. The new config section's values are semantically sensible; torchvision reports invalid ranges or strengths at construction. The default output path `runs/augmentation_preview.png` is appropriate for an ad hoc preview; use `--output` to keep a figure with a specific run.
## Edge cases
- Handled: `--n` below one is rejected; non-RGB source images are converted to RGB; incomplete final grid rows are hidden cleanly.
- Not handled: malformed crop bounds or unsupported hue settings are passed to torchvision and produce its validation error.
## Not verified
No training job or GPU path was run. The preview test uses generated pixels, not project images. Worker-process seeding behavior in a future training DataLoader was not exercised.
## Changes outside the scope
- `src/ie4483/config.py::AugmentConfig`: required so the strict loader accepts and validates the requested configuration section.
- `README.md`: required documentation for the newly added preview command and config controls.
- `tests/test_transforms.py`: required acceptance coverage for the new behavior.
## How it was verified
`tests/test_transforms.py` passed all three tests: tensor size/dtype and normalization, deterministic disabled transforms, enabled switch pipeline differences, and command-line PNG output. Ruff passed for the changed Python files. The full repository suite was started twice but the execution tool returned after eight progress dots without a summary or exit code; its final status is therefore not established and must be rerun by the supervisor.
## To change this
To use a stronger crop, enable `augment.random_resized_crop` and lower `augment.random_resized_crop_scale[0]`; this may remove useful subject context. To compare a specific operation, toggle only its switch while leaving all other values fixed. To change normalization, update `IMAGENET_MEAN` and `IMAGENET_STD` together; changing only one will distort input channel scaling.
## Rollback
Remove `src/ie4483/transforms.py`, `scripts/preview_augment.py`, `tests/test_transforms.py`, the `augment` defaults, the typed augment schema, and the README command together. No existing training caller was present in this pipeline-groundwork phase.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:21 UTC for issue #5 attempt 5; compared HEAD..working tree (base 8d1024d). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +10/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +55/-0 lines
- `src/ie4483/`: 2 file(s), 3 component(s), +100/-2 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/preview_augment.py`, `src/ie4483/transforms.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/transforms.py`, `scripts/preview_augment.py`, `configs/base.yaml`. Verdict: violation.
- out of scope: `src/ie4483/config.py::AugmentConfig` (added); explained in the record: yes
- out of scope: `src/ie4483/config.py::Config` (modified); explained in the record: no
- out of scope: `src/ie4483/config.py::_construct` (modified); explained in the record: no

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

6 checked by the tool, 1 tested in this run, 17 claimed only (of 24 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 6 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 6 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/transforms.py::build_eval_transform` — eval pipeline (named code exists and changed)
  - [tested] How it was verified: `tests/test_transforms.py` passed all three tests: tensor size/dtype and normalization,... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/transforms.py::build_train_transform` — seeded training pipeline (named code exists and changed)
  - [checked] Components: `src/ie4483/transforms.py::describe_transforms` — readable listing (named code exists and changed)
  - [checked] Components: `src/ie4483/config.py::AugmentConfig` — typed controls (named code exists and changed)
  - [checked] Components: `scripts/preview_augment.py::main` — PNG grid command (named code exists and changed)
  - [checked] Components: `configs/base.yaml` — defaults (named code exists and changed)
  - [claimed] Components: `tests/test_transforms.py` — output, switches, preview (names no code)
  - [claimed] Key parameters: `image_size: 224` (`configs/base.yaml::data.image_size`) sets the final square crop; la... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: `--n` below one is rejected; non-RGB source images are converted to RGB; incom... (names no code)
  - full per-claim list (24) in the JSON record

**Consistency**

0 error(s), 3 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won
- WARN: out-of-scope change not explained in 'Changes outside the scope': src/ie4483/config.py::Config
- WARN: out-of-scope change not explained in 'Changes outside the scope': src/ie4483/config.py::_construct

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0005-preprocessing-augmentation.md` gives the commit.
<!-- verified-facts:end -->
