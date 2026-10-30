# 0017: Repeatability check
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #17 · Date: 2026-10-19
## In short
Training now defaults to deterministic torch settings, records the switch in each resolved run config, and exposes a three-run repeatability check. The check compares every epoch’s train/validation loss and accuracy for two runs at the configured seed against a tolerance, then reports the difference from a run at the next seed.
## How it works
1. The owner starts the command with a YAML config, an epoch count, and optionally normal training overrides, an output folder, and a tolerance.
2. The command loads the resolved training settings, changes the epoch count to the requested value, and invokes the existing trainer three times. The first two runs share the configured seed; the last uses the next integer seed. Each gets its own ordinary run folder and saved config.
3. Before model and loader construction, each training run seeds Python, NumPy, and torch. With deterministic mode enabled, cuDNN benchmarking is disabled, deterministic algorithms are requested in warning mode, and the cuBLAS workspace setting is applied.
4. After each training completes, the command reads its metrics CSV and compares train loss, train accuracy, validation loss, and validation accuracy for matching epochs. It prints the largest same-seed difference, the largest difference to the other seed, and PASS or FAIL against the tolerance. A differing epoch count in the repeated runs is an error.
5. The CPU test builds a tiny image-folder dataset, runs all three trainings without pretrained weights, checks exact same-seed equality and a different-seed metric, and checks the saved seed and deterministic setting.
## Main design choice
The checker calls `src/ie4483/train.py::train` directly and uses its metrics CSV as the comparison source. A subprocess runner could isolate failures and global RNG state more strongly, but it would need to duplicate config overrides and coordinate parsing and run artifacts. The existing trainer already writes resolved configs and per-epoch metrics, so those outputs provide a direct check of the behavior the project intends to reproduce. A zero default tolerance makes exact CPU equality the normal expectation; owners can set a nonzero threshold for hardware or backend differences.
## Traps: why the obvious fix is wrong
- Comparing only final validation accuracy can miss divergence in loss or in earlier epochs; all four epoch metrics are compared.
- Reusing one output folder for every repetition can confuse run artifacts; each training call receives a separate subfolder.
- `train.deterministic` must be a typed config field. Keeping it only as a trainer extra causes the normal config reader to reject the key and omits it from the typed resolved configuration.
- YAML treats bare `off` as a boolean in overrides. String-valued training options such as feature-cache mode need quoting when overridden.
- The next seed is observable in the saved config; typed config values must not be overwritten by duplicated raw YAML fields while assembling trainer extras.
## Components
- `src/ie4483/seed.py::seed_everything` — seeding and backend flags
- `src/ie4483/train.py::train` — per-run seeding and artifacts
- `scripts/check_repeatable.py::check_repeatable` — three-run comparison
- `tests/test_train.py::test_repeatability_check_compares_same_seed_and_records_switch` — generated-data check
- `tests/test_seed.py::test_seed_everything_configures_determinism` — backend settings
## Breaks or needs follow-up
None.
## Problem and constraints
Multi-seed comparisons need evidence that one seed produces repeatable metrics. The check must use the project training path and local generated images; it must not download data or weights and must run on CPU.
## Alternatives considered
- Add a separate metrics artifact format: not needed because `metrics.csv` already contains epoch values and is saved in every run folder.
- Compare whole CSV rows: elapsed seconds vary between equivalent runs and do not measure model repeatability, so the checker compares only losses and accuracies.
## Trade-offs
The command performs three complete trainings and therefore costs roughly three run times and disk space. Equality of recorded metrics does not prove that checkpoints are bitwise identical or that a different seed always changes every metric. The report includes the different-seed difference as a sanity signal; only the same-seed tolerance controls the exit result.
## Key parameters
- `train.deterministic=true` (`configs/base.yaml::train.deterministic`): default enables deterministic algorithms with warnings; false disables the request and enables cuDNN benchmarking.
- `--epochs N` (`scripts/check_repeatable.py::main`): required positive epoch count; raising it extends each run and the compared metric series.
- `--tolerance` (`scripts/check_repeatable.py::main`): default `0.0`; raising it accepts larger same-seed differences, lowering it tightens the equality condition.
- `train.seed` (`configs/base.yaml::train.seed`): seed used in both repeat runs; the comparison run uses seed + 1.
## Assumptions
Both repeated runs see the same data, configuration, software, and device. PyTorch operations either honor deterministic settings or emit warnings where deterministic behavior is not guaranteed.
## Edge cases
- Handled: rejects epoch counts below one and negative tolerances; detects differing same-seed epoch counts.
- Not handled: if the third run has no overlapping epochs with the first, the reported different-seed difference has no meaningful comparison; the same-seed check remains the gating result.
## Not verified
GPU repeatability and behavior of every CUDA operation with the selected PyTorch build were not measured. The CPU fixture does not establish that different seeds always produce different metrics for every dataset or model.
## Changes outside the scope
- `src/ie4483/config.py::TrainConfig` and `configs/base.yaml::train`: required typed default for `train.deterministic`, so the requested key loads and is saved in run configs.
- `README.md`: required usage instructions for the new command and config key.
- `tests/test_seed.py` and `tests/test_train.py`: required behavior coverage.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_seed.py tests/test_train.py` passed 9 tests. The full CPU suite, `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q`, passed 67 tests in 132.63 seconds. Ruff passed on the changed Python files. `git diff --check` completed without whitespace errors; Git emitted environment permission warnings while reading the user-level ignore file and pytest cache.
## To change this
To allow small backend differences, pass a larger `--tolerance`; the cost is accepting that much same-seed metric drift. To compare other metrics, change `scripts/check_repeatable.py::METRIC_COLUMNS`; including elapsed time would make equivalent runs appear different.
## Rollback
Remove `scripts/check_repeatable.py`, its README instructions and tests, and the deterministic config field. Restore `src/ie4483/seed.py::seed_everything` to its previous one-argument behavior and remove the deterministic call from `src/ie4483/train.py`. Existing run folders retain their saved config and metrics; no migration is needed.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:35 UTC for issue #17 attempt 4; compared HEAD..working tree (base 32860bc). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +1/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +68/-0 lines
- `src/ie4483/`: 3 file(s), 4 component(s), +16/-6 lines
- plus 2 test file(s), 1 doc file(s)
- new files: `scripts/check_repeatable.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/seed.py`, `scripts/check_repeatable.py`, `src/ie4483/train.py`. Verdict: violation.
- out of scope: `configs/base.yaml` (modified); explained in the record: yes
- out of scope: `src/ie4483/config.py::TrainConfig` (modified); explained in the record: yes

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

3 checked by the tool, 0 tested in this run, 16 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 3 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/seed.py::seed_everything` — seeding and backend flags (named code exists and changed)
  - [checked] Components: `src/ie4483/train.py::train` — per-run seeding and artifacts (named code exists and changed)
  - [checked] Components: `scripts/check_repeatable.py::check_repeatable` — three-run comparison (named code exists and changed)
  - [claimed] Components: `tests/test_train.py::test_repeatability_check_compares_same_seed_and_records_switch` —... (names no code)
  - [claimed] Key parameters: `train.deterministic=true` (`configs/base.yaml::train.deterministic`): default enables... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: rejects epoch counts below one and negative tolerances; detects differing same... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_seed.py tests/t... (no test named)
  - [claimed] Components: `tests/test_seed.py::test_seed_everything_configures_determinism` — backend settings (names no code)
  - [claimed] Key parameters: `--epochs N` (`scripts/check_repeatable.py::main`): required positive epoch count; rais... (no cited constant with a matching value)
  - [claimed] Key parameters: `--tolerance` (`scripts/check_repeatable.py::main`): default `0.0`; raising it accepts... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won
- WARN: references something that doesn't exist: metrics.csv

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0017-repeatability-check.md` gives the commit.
<!-- verified-facts:end -->
