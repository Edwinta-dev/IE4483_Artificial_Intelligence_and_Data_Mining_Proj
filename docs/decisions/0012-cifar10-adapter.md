# 0012: CIFAR-10 adapter, configs and test evaluation
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #12 · Date: 2026-10-19
## In short
CIFAR-10's Python batches now feed the existing transfer-learning loop, with validation carved deterministically from training data and the official test batch reserved for final evaluation. The run configuration selects a 10-class head, larger inputs and CIFAR normalization; evaluation saves class metrics and the confusion matrix with the run artifacts.
## How it works
1. The owner runs the one-off fetch command to place the upstream archive under the configured data root and extract the `cifar-10-batches-py` directory. Tests create local fake batches and do not call this command.
2. Training reads all five training batches, converts the packed channel-first bytes into RGB images and divides each class's examples into a training portion and a validation portion using the configured training seed. The official `test_batch` is not read during this step.
3. Each image is enlarged to the configured square input size and normalized with CIFAR channel statistics. Training also applies a padded random crop and random horizontal flip. The existing seed setup controls the random split, training loader shuffle and augmentation.
4. The existing model, optimizer, validation-based checkpoint choice and training loop run unchanged in their optimization behavior. A ten-entry class list configures the output head, and CIFAR augmentation prevents reuse of the image-folder frozen-feature cache.
5. Evaluation defaults to validation. Selecting the test split loads the official test batch only after a checkpoint exists, then writes split-specific predictions and metrics, a confusion-matrix image, and the normal latest-result summary. The metric JSON includes a per-class accuracy, precision, recall and F1 value plus the full confusion matrix.
## Main design choice
`src/ie4483/cifar.py::CIFAR10Dataset` reads the standard batch serialization directly and implements the seeded split without copying data into a temporary folder tree. Reusing `src/ie4483/data.py::ImageFolderDataset` would require expanding 50,000 small images to disk and inventing folder and manifest conventions for array data; that would complicate deterministic splitting and consume extra storage. A torchvision dataset was another plausible option, but it can trigger implicit downloads or use a different cache layout, conflicting with the local-data-only requirement. To reuse the project's training loop and make `--split test` available, the minimal unavoidable integration changes touch the training and evaluation modules, typed model configuration and evaluator CLI outside the issue's Scope line. Extending the typed model config with `num_classes` was selected over inferring the classifier size from class-list length inside the model factory; explicit config preserves the model head contract and keeps binary configurations at two outputs.
## Traps: why the obvious fix is wrong
- Do not use the official test batch to carve validation or select a checkpoint: the reported test number would no longer be a final held-out result.
- Batch images are stored as flattened channel-first bytes. Treating them as already interleaved RGB gives incorrect color channels while preserving a plausible image shape.
- CIFAR labels are integer IDs in the order listed in `src/ie4483/cifar.py::CIFAR10_CLASSES`; alphabetical sorting would silently remap targets and reported class names.
- Do not use the folder feature cache for CIFAR training: random crop and flip mean that cached features would freeze one augmentation view and no longer match the training input path.
- Keep the established validation confusion-matrix filename for existing callers. The test output has its own filename so evaluation does not overwrite validation artifacts.
## Components
- `src/ie4483/cifar.py::CIFAR10Dataset` — batches and split
- `src/ie4483/cifar.py::cifar_transform` — resize, augmentation, normalization
- `src/ie4483/train.py::train` — CIFAR dataset selection and cache bypass
- `src/ie4483/evaluate.py::evaluate` — validation/test metrics and artifacts
- `scripts/fetch_cifar10.py::main` — owner download command
- `configs/cifar10.yaml` — 10-class run
- `tests/test_cifar.py` — generated batch integration
## Breaks or needs follow-up
None. Existing validation artifact names and the default validation evaluation behavior remain intact.
## Problem and constraints
The issue needs the existing ResNet transfer-learning pipeline to cover standard CIFAR-10 batches, use validation for model selection, and report the official test set without network access in tests or dependence on a GPU. Data and generated artifacts remain outside the repository.
## Alternatives considered
The folder dataset route would duplicate CIFAR into image files and lose the direct batch representation. The torchvision loader was rejected because its download behavior and storage conventions weaken local data-root control. A specialized CIFAR training loop was rejected because the existing loop already supports arbitrary class counts and validation-based checkpoint selection.
## Trade-offs
The adapter keeps the original CIFAR arrays resident in memory and makes a copied selected array for each split, trading RAM for simple random access and no intermediate image files. Per-class stratification provides class coverage for small fake datasets, though the per-class validation counts can round differently for tiny or uneven batches. Evaluation currently retains predictions until it writes outputs, matching the existing evaluator's memory use.
## Key parameters
- Input size 96 (`configs/cifar10.yaml::data.image_size`): gives ResNet larger spatial input; increasing it raises compute and memory, decreasing it loses detail.
- Ten classes (`configs/cifar10.yaml::data.classes`, `configs/cifar10.yaml::model.num_classes`): defines label names and head width; both must remain aligned with IDs 0 through 9.
- CIFAR mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)` (`src/ie4483/cifar.py::CIFAR10_MEAN`, `src/ie4483/cifar.py::CIFAR10_STD`): normalize the dataset channel distribution; changing these changes model input scale.
- Validation fraction 0.1 (`src/ie4483/cifar.py::CIFAR10Dataset`): holds out one tenth per class by default; a larger fraction reduces training examples.
- Split seed 42 (`configs/cifar10.yaml::train.seed`): determines validation membership and training shuffle; changing it creates another split/run.
- Batch size 32 and evaluation batch size 64 (`configs/cifar10.yaml::train.batch_size`, `configs/cifar10.yaml::eval.batch_size`): control memory and throughput.
- Crop padding 4 pixels and flip probability 0.5 (`src/ie4483/cifar.py::cifar_transform` and torchvision `RandomCrop`/`RandomHorizontalFlip` defaults): increase crop and orientation variation; stronger augmentation may discard more edge content.
## Assumptions
The local archive uses the standard CIFAR-10 Python keys and 32 by 32 RGB data encoding; an incompatible or corrupt batch raises during loading. The config's class order matches official integer labels. The upstream archive's availability and extraction on the owner's machine were not exercised here.
## Edge cases
 - Handled: unknown split names and invalid validation fractions raise clear errors; split sizes are selected independently by class; zero readable labels cause evaluation to raise instead of producing misleading metrics.
 - Not handled: damaged CIFAR files are not skipped or repaired; loading reports the pickle/shape error to the caller.
## Not verified
The owner download command was not run and no real dataset or pretrained weights were used. The recorded test metrics validate output shape and wiring only; they are not model performance claims.
## Changes outside the scope
- `src/ie4483/config.py::ModelConfig`: adds explicit `num_classes`, required for typed CIFAR YAML head configuration.
- `src/ie4483/train.py::train`: selects CIFAR data/transforms and bypasses image-folder feature caching for randomized CIFAR views.
- `src/ie4483/evaluate.py::evaluate` and `scripts/evaluate.py::main`: add official test-split evaluation, confusion-matrix JSON, and `--split` support.
- `README.md`: required usage and dataset-difference documentation because this change adds a command and config.
- `tests/test_cifar.py`: required generated-batch, CPU training, and test evaluation acceptance coverage.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_cifar.py` passed both CIFAR tests. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_evaluate.py tests/test_cifar.py` passed three tests. Final `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 47 tests in 73.54 seconds. Ruff passed on changed Python modules, CLI scripts and CIFAR tests.
## To change this
To use a different held-out share, change the `val_fraction` argument/default at `src/ie4483/cifar.py::CIFAR10Dataset`; the config currently leaves it at 0.1. To alter input resolution, change `configs/cifar10.yaml::data.image_size`, with compute and memory costs scaling upward. To change augmentation, edit `src/ie4483/cifar.py::cifar_transform`; disabling both random operations also requires revisiting the CIFAR cache bypass in `src/ie4483/train.py::train`.
## Rollback
Remove the CIFAR adapter, YAML, fetch script and generated-batch tests, then revert the CIFAR branches in training/evaluation, the `num_classes` schema field and README section. Before reverting evaluation integration, confirm no CIFAR run depends on `--split test`; Dogs vs Cats validation continues to use the unchanged default path.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 03:17 UTC for issue #12 attempt 12; compared HEAD..working tree (base 4c002e4). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +19/-0 lines
- `scripts/`: 2 file(s), 1 component(s), +30/-1 lines
- `src/ie4483/`: 4 file(s), 4 component(s), +116/-21 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `configs/cifar10.yaml`, `scripts/fetch_cifar10.py`, `src/ie4483/cifar.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/cifar.py`, `src/ie4483/data.py`, `configs/cifar10*.yaml`, `scripts/fetch_cifar10.py`. Verdict: violation.
- out of scope: `scripts/evaluate.py::main` (modified); explained in the record: yes
- out of scope: `src/ie4483/config.py::ModelConfig` (modified); explained in the record: yes
- out of scope: `src/ie4483/evaluate.py::_scores` (modified); explained in the record: no
- out of scope: `src/ie4483/evaluate.py::evaluate` (modified); explained in the record: yes
- out of scope: `src/ie4483/train.py::train` (modified); explained in the record: yes

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

6 checked by the tool, 0 tested in this run, 18 claimed only (of 24 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 6 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 7 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/cifar.py::CIFAR10Dataset` — batches and split (named code exists and changed)
  - [checked] Components: `src/ie4483/cifar.py::cifar_transform` — resize, augmentation, normalization (named code exists and changed)
  - [checked] Components: `src/ie4483/train.py::train` — CIFAR dataset selection and cache bypass (named code exists and changed)
  - [checked] Components: `src/ie4483/evaluate.py::evaluate` — validation/test metrics and artifacts (named code exists and changed)
  - [checked] Components: `scripts/fetch_cifar10.py::main` — owner download command (named code exists and changed)
  - [checked] Components: `configs/cifar10.yaml` — 10-class run (named code exists and changed)
  - [claimed] Components: `tests/test_cifar.py` — generated batch integration (names no code)
  - [claimed] Key parameters: Input size 96 (`configs/cifar10.yaml::data.image_size`): gives ResNet larger spatial in... (no cited constant with a matching value)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_cifar.py` passe... (no test named)
  - [claimed] Key parameters: Ten classes (`configs/cifar10.yaml::data.classes`, `configs/cifar10.yaml::model.num_cla... (no cited constant with a matching value)
  - full per-claim list (24) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: out-of-scope change not explained in 'Changes outside the scope': src/ie4483/evaluate.py::_scores

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0012-cifar10-adapter.md` gives the commit.
<!-- verified-facts:end -->
