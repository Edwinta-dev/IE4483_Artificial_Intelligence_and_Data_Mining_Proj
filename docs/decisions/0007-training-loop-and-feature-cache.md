# 0007: Training loop and frozen feature cache
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #7 · Date: 2026-10-19
## In short
The pipeline now has a seeded ResNet training command with optimizer groups, schedules, optional AMP, early stopping, best-accuracy checkpointing, and per-epoch metrics. Frozen runs without augmentation can cache pooled train and validation features under a content-derived key.
## How it works
1. The command reads YAML settings, applies dotted overrides, validates the established typed configuration, and combines it with training controls handled by the training module. It creates a uniquely named run folder.
2. It writes resolved settings, a readable transform description, and the selected training-file manifest. Training uses the configured train transform; validation uses the evaluation transform.
3. When the backbone is frozen, augmentation is disabled, and caching is auto, the program hashes the backbone choice, pretrained setting, image size, evaluation-transform identity, and the paths and labels for both splits. It computes pooled features once or loads the matching cache, then trains only the head. Other configurations use normal image batches.
4. Training uses cross-entropy, a configurable optimizer and schedule, and CUDA autocasting only when CUDA is available and enabled. CPU stays in fp32. Each epoch records train and validation loss and accuracy, learning rate, and elapsed time.
5. Validation loss controls patience; a checkpoint is replaced only when validation accuracy improves. Run artifacts and reusable features stay under the run root.
## Main design choice
The training loop consumes the existing typed config for established project fields and reads issue-specific controls from YAML in the in-scope module. Extending the typed schema in src/ie4483/config.py would provide centralized validation, but that file is outside the declared Scope. The in-scope loader filters its own fields before calling the strict existing loader. For caching, pooled features and labels were chosen over decoded-image caching or repeated backbone inference: pooled vectors remove the expensive image path from repeated head sweeps while retaining the classifier input. This only benefits frozen, unaugmented experiments; trainable weights or random transforms make cached features stale.
## Traps: why the obvious fix is wrong
- Frozen parameters alone do not make features reusable when augmentation changes pixels; cache eligibility checks augmentation switches.
- Cache keys include the manifest and image size because either changing under a stable filename would mix old features with new data.
- Early stopping follows validation loss while checkpoint selection follows validation accuracy. Combining them changes a requested model-selection criterion.
- AMP is enabled only for CUDA; CPU must remain fp32.
- Cached training still saves the full model state, including the frozen backbone; head-only optimization must not produce an incomplete checkpoint.
## Components
- src/ie4483/train.py::train — orchestration and artifacts
- src/ie4483/train.py::_fit — optimization, metrics, stopping, checkpoint
- src/ie4483/train.py::load_training_config — config and CLI overrides
- src/ie4483/features.py::feature_cache_key — feature identity
- src/ie4483/features.py::extract_features — frozen pooling pass
- src/ie4483/features.py::load_or_create_cache — reuse and persistence
- scripts/train.py::main — command entry point
- tests/test_train.py — run and cache behavior
- configs/base.yaml — default VRAM batch note
## Breaks or needs follow-up
None known. The typed Config dataclasses do not expose the new issue-specific training fields; the command accepts and logs them through its own loader. A later typed-config issue could move these controls into the dataclasses.
## Problem and constraints
Rubric parts (b) and (c) need an explicit and repeatable recipe with logs to inspect decisions. Work stays in the declared training, feature, command, and config areas; tests use generated images and never fetch weights or data. The test path forces CPU even if the host reports CUDA.
## Alternatives considered
- Add settings to src/ie4483/config.py: clearer schema, but outside this issue's Scope.
- Save image tensors rather than pooled vectors: preprocessing is reusable, but backbone inference is still paid on each run.
- Cache regardless of augmentation or freeze mode: invalid for changing images or weights.
## Trade-offs
Cache creation adds one pass and disk storage; for small runs, direct image training can be faster. Metrics flush per epoch, so interruption mid-epoch leaves no partial row. Cached runs collect features in host memory before saving, which may matter for much larger datasets.
## Key parameters
- Batch size 32 (configs/base.yaml::train.batch_size): intended for frozen ResNet-50 at 224 on 4 GB VRAM; use 4 for fine tuning and accumulation to recover effective batch size.
- AdamW; head LR 0.001 (configs/base.yaml::train.learning_rate); backbone LR 0.0001 (src/ie4483/train.py::_EXTRA_DEFAULTS): lower backbone LR limits large updates to transferred features.
- Cosine schedule; 10 epochs (src/ie4483/train.py::_EXTRA_DEFAULTS, configs/base.yaml::train.epochs): cosine spans the configured run. Step schedule uses step size 5 and gamma 0.1.
- Patience 3 (src/ie4483/train.py::_EXTRA_DEFAULTS): stops after more than three validation-loss non-improvements.
- Cache auto (src/ie4483/train.py::_EXTRA_DEFAULTS): off skips persistence; auto requires frozen mode and disabled augmentation.
- Label smoothing 0.0, weight decay 0.01, momentum 0.9, warmup 0, workers 0 (src/ie4483/train.py::_EXTRA_DEFAULTS): defaults avoid smoothing and warmup; weight decay regularizes; workers control loader parallelism.
## Assumptions
The pretrained source is the model factory's default cached torchvision checkpoint, so the pretrained boolean identifies the source. Cache files are local outputs. The model wrapper exposes its ResNet backbone and head as currently implemented.
## Edge cases
- Handled: invalid optimizer, schedule, and cache mode raise value errors; cache replacement uses a temporary file before rename.
- Not handled: corrupt existing cache fails to load and must be removed; invalid class weights are delegated to PyTorch loss validation.
## Not verified
CUDA AMP and 4 GB VRAM behavior were not exercised. No real dataset or pretrained checkpoint was used. The default batch guidance was not benchmarked in this implementation.
## Changes outside the scope
- README.md: documents the added command and settings, as required by repository rules.
- tests/test_train.py: required behavior coverage.
These are required supporting changes. No production module outside the declared Scope was changed.
## How it was verified
tests/test_train.py passed all four tests, covering generated-image CPU training, run artifacts, finite losses, seeded metric equality, cache reuse without a second extraction, key changes for size and manifest, one-time cache-builder reuse, patience-zero stopping, and cached-feature/head output equivalence. Ruff passed for all changed Python files. The full CPU suite passed 34 tests in 53.42 seconds.
## To change this
- To use SGD, set train.optimizer=sgd; tune momentum and weight decay.
- To use step decay, set train.scheduler=step and adjust train.step_size or train.gamma.
- To reduce GPU memory, lower train.batch_size from 32 to 4 for fine tuning and raise train.gradient_accumulation_steps; accumulation costs time.
- To avoid cache storage, set train.feature_cache=off; repeated frozen runs recompute features.
## Rollback
Remove src/ie4483/train.py, src/ie4483/features.py, scripts/train.py, tests/test_train.py, revert the README and base-config note, and remove this record. No earlier production component imports the training module.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:35 UTC for issue #7 attempt 7; compared HEAD..working tree (base b919d16). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +1/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +22/-0 lines
- `src/ie4483/`: 2 file(s), 0 component(s), +283/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/train.py`, `src/ie4483/features.py`, `src/ie4483/train.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/train.py`, `src/ie4483/features.py`, `scripts/train.py`, `configs/**`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

7 checked by the tool, 0 tested in this run, 19 claimed only (of 26 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 7 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 6 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: src/ie4483/train.py::train — orchestration and artifacts (named code exists and changed)
  - [checked] Components: src/ie4483/train.py::_fit — optimization, metrics, stopping, checkpoint (named code exists and changed)
  - [checked] Components: src/ie4483/train.py::load_training_config — config and CLI overrides (named code exists and changed)
  - [checked] Components: src/ie4483/features.py::feature_cache_key — feature identity (named code exists and changed)
  - [checked] Components: src/ie4483/features.py::extract_features — frozen pooling pass (named code exists and changed)
  - [checked] Components: src/ie4483/features.py::load_or_create_cache — reuse and persistence (named code exists and changed)
  - [checked] Components: scripts/train.py::main — command entry point (named code exists and changed)
  - [claimed] Components: tests/test_train.py — run and cache behavior (names no code)
  - [claimed] Key parameters: Batch size 32 (configs/base.yaml::train.batch_size): intended for frozen ResNet-50 at 2... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: invalid optimizer, schedule, and cache mode raise value errors; cache replacem... (names no code)
  - full per-claim list (26) in the JSON record

**Consistency**

0 error(s), 0 warning(s) between the rationale above and the change.

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0007-training-loop-and-feature-cache.md` gives the commit.
<!-- verified-facts:end -->
