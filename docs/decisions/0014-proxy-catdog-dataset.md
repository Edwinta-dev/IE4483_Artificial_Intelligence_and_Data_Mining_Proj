# 0014: CIFAR-10 cat and dog proxy dataset
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #14 · Date: 2026-10-19
## In short
A local converter now turns CIFAR-10 cat and dog examples into the same class-folder layout used by the Dogs vs Cats pipeline. It marks the output as a stand-in, records source identities and hashes, and marks validation metrics so proxy scores cannot be mistaken for the project result.
## How it works
1. The owner fetches CIFAR-10 with the existing fetch command, then points the converter at the extracted data root and an output folder.
2. The converter opens CIFAR’s Python batches through the existing reader. It uses that reader’s seeded training/validation split, selects the requested number of cat and dog examples per class, and saves their original 32-pixel RGB images (`src/ie4483/cifar.py::_batch`) as PNGs in the binary folder layout.
3. It takes only CIFAR’s official test batch for the proxy’s unlabeled test tree. Those cat and dog images receive ascending numeric filenames, without class names in the test folder.
4. A CSV manifest keeps each output tied to its split, class when labeled, source batch index, and pixel hash. The marker describes the dataset as a stand-in and states that its scores are not Dogs vs Cats results.
5. The standard folder trainer, evaluator, and predictor can then consume the output. Evaluation checks for the marker and adds `proxy: true` to the run’s validation metrics. The config uses the existing binary ResNet-18 path and the runbook shows the full owner workflow.
## Main design choice
Materialize the selected images into the project’s existing folder contract and reuse `src/ie4483/cifar.py::CIFAR10Dataset` for the deterministic split. A direct CIFAR binary training adapter was a plausible alternative, but would not exercise the folder loader, audit command, or normal prediction path that this stand-in is intended to validate. The trade-off is extra local disk writes and PNG decoding. CIFAR’s official test batch remains isolated as unlabeled proxy test data; deriving validation from it would invalidate the held-out evaluation convention already established by `src/ie4483/cifar.py::CIFAR10Dataset`.
## Traps: why the obvious fix is wrong
- Do not call the proxy a cleaned Dogs vs Cats dataset or report its metrics as the project result; CIFAR is only a pipeline stand-in.
- Do not build validation from `test_batch`; the existing seeded split already separates train and validation within the training batches.
- The output test files intentionally have no class subfolders or labels. Their numeric IDs follow the submission contract and preserve the unlabeled prediction path.
- The output directory must be empty or absent. Reusing a nonempty destination could leave stale images that invalidate counts and overlap checks.
- `n_train` and `n_val` mean per-class counts, so totals are twice those values for the two selected classes.
## Components
- `scripts/make_proxy_catdog.py::make_proxy` — materialization and manifest
- `scripts/make_proxy_catdog.py::main` — command line
- `src/ie4483/proxy.py::is_proxy_dataset` — marker detection
- `src/ie4483/proxy.py::write_proxy_marker` — marker metadata
- `src/ie4483/proxy.py::proxy_metric` — metrics flag
- `src/ie4483/evaluate.py::evaluate` — metric integration
- `configs/proxy_catdog.yaml` — binary proxy training settings
- `tests/test_proxy.py` — generated CIFAR integration coverage
- `README.md` — owner workflow and reporting note
## Breaks or needs follow-up
None. Existing run metrics gain one boolean field only when the configured data root contains the proxy marker.
## Problem and constraints
The cleaned project images are not ready, so the pipeline needs a repeatable local stand-in with the same `train/{cat,dog}`, `val/{cat,dog}`, and numeric `test/` contract. Tests generate tiny CIFAR-format batches in temporary storage and must work without network access, downloaded weights, or CUDA. The converter itself only reads local files.
## Alternatives considered
- Copy all CIFAR classes — would add unrelated classes and require a different classifier head.
- Create validation from `test_batch` — would reuse a held-out split for model selection and contradict the CIFAR adapter decision.
- Add a new image-folder exporter with its own split algorithm — would duplicate the seeded split logic already implemented in `src/ie4483/cifar.py::CIFAR10Dataset`.
## Trade-offs
Materializing PNGs uses storage and file operations, and resized training at 224 pixels costs more CPU/GPU time than the native 32-pixel data. This is acceptable for a small binary proxy and verifies the actual folder path. The generator rejects insufficient class counts instead of silently producing smaller datasets.
## Key parameters
- 4,000 training images per class, `scripts/make_proxy_catdog.py::main` argument default: matches the available CIFAR training scale; lowering it shortens proxy runs, raising it requires enough class examples after the seeded split.
- 400 validation images per class, `scripts/make_proxy_catdog.py::main` argument default: gives a nontrivial binary validation set; lower counts make metrics noisier and higher counts leave fewer training examples.
- Seed 42, `scripts/make_proxy_catdog.py::main` argument default: selects a stable split and file selection; changing it changes selected examples.
- 224-pixel input, `configs/proxy_catdog.yaml::data.image_size`: matches the Dogs vs Cats configuration; reducing it speeds runs but changes the intended preprocessing workload.
- Batch size 4 and accumulation 8, `configs/proxy_catdog.yaml::train`: conservative settings for the target 4 GB GPU; increasing batch size raises memory use.
## Assumptions
The user has already fetched and extracted standard CIFAR-10 Python batches under the supplied root. The converter expects the canonical batch format read by `src/ie4483/cifar.py::_batch`; alternate serialization is rejected by the batch reader.
## Edge cases
- Handled: missing requested class count, nonpositive requested count, absent or nonempty output directory, reproducible selection, marker and manifest creation.
- Not handled: cleanup of a partially written output if conversion fails midway; the user sees a partial directory and must remove it before retrying.
## Not verified
The owner download workflow and GPU path were not run. Tests use generated CIFAR batches, CPU, and `pretrained: false`; no CIFAR benchmark scores were measured.
## Changes outside the scope
- `src/ie4483/evaluate.py::evaluate`: required to attach the marker-derived `proxy` field to the promised `val_metrics.json`; the Scope line names the proxy module, configs, script, and README but omits the existing evaluator entry point.
## How it was verified
`tests/test_proxy.py` creates fake CIFAR batch files, invokes the converter twice with the same seed, checks paths/counts and selection stability, verifies source-index and hash separation, runs `scripts/check_setup.py`, checks `src/ie4483/data_audit.py::audit_data`, trains one epoch on CPU, evaluates marked validation data, and runs prediction. `tests/test_proxy.py` passed; `ruff check` passed for the changed Python files; the complete CPU suite passed (55 tests in 150.91 seconds).
## To change this
- To change generated split sizes, set `--n-train` or `--n-val`; each controls images per class.
- To select a different training split, set `--seed`; the cost is a different sample set and results.
- To reduce proxy training cost, lower `configs/proxy_catdog.yaml::data.image_size` or `configs/proxy_catdog.yaml::train.epochs`; the result then exercises a smaller workload.
## Rollback
Remove the proxy converter, marker helper, proxy config, proxy test, README subsection, and this record. Remove the metric import and update in `src/ie4483/evaluate.py::evaluate` to restore the previous validation-metric schema; no existing configuration depends on proxy markers.


<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:00 UTC for issue #14 attempt 1; compared HEAD..working tree (base f463ce6). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +22/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +74/-0 lines
- `src/ie4483/`: 2 file(s), 1 component(s), +30/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `configs/proxy_catdog.yaml`, `scripts/make_proxy_catdog.py`, `src/ie4483/proxy.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/make_proxy_catdog.py`, `src/ie4483/proxy.py`, `configs/proxy_catdog*.yaml`, `README.md`. Verdict: violation.
- out of scope: `src/ie4483/evaluate.py::evaluate` (modified); explained in the record: yes

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

7 checked by the tool, 1 tested in this run, 17 claimed only (of 25 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 7 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 5 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `scripts/make_proxy_catdog.py::make_proxy` — materialization and manifest (named code exists and changed)
  - [tested] How it was verified: `tests/test_proxy.py` creates fake CIFAR batch files, invokes the converter twice with... (test gate passed: project test suite (CPU))
  - [checked] Components: `scripts/make_proxy_catdog.py::main` — command line (named code exists and changed)
  - [checked] Components: `src/ie4483/proxy.py::is_proxy_dataset` — marker detection (named code exists and changed)
  - [checked] Components: `src/ie4483/proxy.py::write_proxy_marker` — marker metadata (named code exists and changed)
  - [checked] Components: `src/ie4483/proxy.py::proxy_metric` — metrics flag (named code exists and changed)
  - [checked] Components: `src/ie4483/evaluate.py::evaluate` — metric integration (named code exists and changed)
  - [checked] Components: `configs/proxy_catdog.yaml` — binary proxy training settings (named code exists and changed)
  - [claimed] Components: `tests/test_proxy.py` — generated CIFAR integration coverage (names no code)
  - [claimed] Key parameters: 4,000 training images per class, `scripts/make_proxy_catdog.py::main` argument default:... (no cited constant with a matching value)
  - full per-claim list (25) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: val_metrics.json

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0014-proxy-catdog-dataset.md` gives the commit.
<!-- verified-facts:end -->
