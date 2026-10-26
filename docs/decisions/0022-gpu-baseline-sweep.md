# 0022: GPU baseline sweep on real data
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #22 · Date: 2026-10-26
## In short
The six completed real-data ResNet-18/50 x freeze-mode runs are summarized in `docs/sweep_results.md`. ResNet-50 frozen reached the highest validation accuracy and was selected as the practical baseline because its epoch time is far lower than fine tuning.
## How it works
1. The owner runs the saved sweep configuration after preparing the real image folders and local pretrained weights; this issue does not start training or fetch anything.
2. The sweep runner makes one run for each requested backbone and freeze-mode pair, with seed 42. Each run records its resolved configuration and per-epoch training and validation measurements in its ignored run folder.
3. A run counts as complete only when validation has written `val_metrics.json`. The report builder checks that file, rejects a proxy marker, and takes the selected validation accuracy from it. It reads the epoch count and time measurements from that run's metrics log.
4. The resulting table compares all six settings and explains the selected baseline. The run folders stay in `runs/`; only the repeatable YAML configuration and the report are tracked.
## Main design choice
The report uses completed run artifacts as its source instead of copying the sweep's aggregate mean table or recomputing evaluation. The existing sweep aggregate has validation accuracy but omits epoch counts and timing; rerunning evaluation could change historical results or require data and weights. Using the saved validation result plus the epoch log directly supplies all requested values while retaining traceability. A generated-on-demand report would avoid a checked-in result, but would not satisfy the requested report artifact for this finished sweep. The config uses batch 71 as specified by the completed run configurations rather than the stale partial value 32.
## Traps: why the obvious fix is wrong
- A `metrics.csv` file alone does not prove validation finished; require `val_metrics.json` as the completion marker.
- A proxy field may be recorded in validation or latest evaluation output; reject either before reporting.
- Do not pick the last epoch's score: the requested best validation accuracy is the final validation result after model selection, recorded in `val_metrics.json`.
- Do not average cell means or infer epoch time from total wall time; use the individual run's `metrics.csv` rows.
- A single seed cannot demonstrate stability across random seeds.
## Components
- `configs/sweeps/gpu_baseline.yaml` — six-cell grid and training settings
- `scripts/sweep.py::write_report` — validated report generation
- `tests/test_experiments.py::test_gpu_report_uses_completed_metrics_and_rejects_proxy` — artifact checks
- `docs/sweep_results.md` — observed results
- `runs/sweeps/gpu_baseline/` — ignored run artifacts
## Breaks or needs follow-up
None.
## Problem and constraints
Issue #22 requires six finished real-data GPU runs, a reproduction config, and a report showing the best validation accuracy, epoch count, and mean time per epoch. Training and downloads are prohibited during implementation; the owner-provided run folders are evidence for the report and are not committed.
## Alternatives considered
- Report only `summary.md`: it lacks timing and epoch count, so it cannot answer the requested comparison.
- Run evaluation again: adds weight/data dependence and may overwrite metrics for a historical run.
- Choose full fine tuning because it trains more parameters: rejected because the observed validation score is lower and time cost is much higher (`docs/sweep_results.md`).
## Trade-offs
The report is a snapshot of one seed and one hardware run. It is reproducible as a record of those artifacts but does not establish variability across seeds or devices.
## Key parameters
- Seed 42: `configs/sweeps/gpu_baseline.yaml::seeds`; fixes stochastic initialization and data ordering for this comparison.
- Batch size 71: `configs/sweeps/gpu_baseline.yaml::overrides`; matches all six completed run configs.
- AMP enabled: `configs/sweeps/gpu_baseline.yaml::overrides`; follows the requested GPU run settings.
- Patience 3 and maximum 10 epochs: `configs/sweeps/gpu_baseline.yaml::overrides`; allow early stopping while bounding training.
- Best accuracy, epoch counts, and seconds per epoch: values in `docs/sweep_results.md`; derived from each run's `val_metrics.json` and `metrics.csv`.
## Assumptions
The six folders in `runs/sweeps/gpu_baseline/` correspond to the requested cleaned Dogs vs Cats data and the owner-run CUDA sweep. Run configs confirm seed 42, batch 71, AMP, and patience 3; validation artifacts do not show a proxy marker. If those folders are replaced, regenerate the report and recheck the provenance.
## Edge cases
- Handled: missing `val_metrics.json` and proxy-marked results raise an error in the report helper.
- Not handled: malformed metric rows fail parsing; investigate the source run rather than manually inserting values.
## Not verified
The report helper's CLI integration was not exercised against a fresh sweep invocation. No additional training was performed, and no second seed or hardware comparison was measured.
## Changes outside the scope
None.
## How it was verified
- Inspected all six run directories and confirmed each has `val_metrics.json`, `metrics.csv`, and a saved resolved config.
- Independently extracted validation accuracy, epoch count, and arithmetic mean of epoch seconds from the six run artifacts; verified every config records seed 42, batch 71, AMP enabled, and patience 3, and no proxy marker.
- `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_experiments.py` passed after the report-helper changes.
- `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 79 tests in 121.44 seconds; one pre-existing scheduler-order warning was reported by `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`.
- Ruff passed for `scripts/sweep.py` and `tests/test_experiments.py`.
## To change this
- To use a different batch, update `train.batch_size` in `configs/sweeps/gpu_baseline.yaml` and rerun all cells; existing folders encode their batch in `config.yaml`.
- To select a speed/accuracy trade-off other than the highest validation score, change the selection rule and explain it in `docs/sweep_results.md`.
## Rollback
Remove the GPU baseline config and README instructions, report, report helper, its tests, and this decision record together. The owner-run folders are ignored artifacts and are not part of rollback.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-26 15:52 UTC for issue #22 attempt 3; compared HEAD..working tree (base 58f2d24). Records are claims to check against the code, not proof.

**Change summary**

- `configs/sweeps/`: 1 file(s), 0 component(s), +11/-0 lines
- `scripts/`: 1 file(s), 3 component(s), +49/-0 lines
- plus 1 test file(s), 2 doc file(s)
- new files: `configs/sweeps/gpu_baseline.yaml`

**Removed symbols**

None.

**Scope**

Declared: `configs/**`, `scripts/sweep.py`, `docs/**`, `README.md`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

2 checked by the tool, 2 tested in this run, 20 claimed only (of 24 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 2 checked, 0 tested, 3 claimed
- Key parameters: 0 checked, 0 tested, 5 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 2 tested, 3 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `configs/sweeps/gpu_baseline.yaml` — six-cell grid and training settings (named code exists and changed)
  - [tested] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 79 tests in 121.44... (test gate passed: project test suite (CPU))
  - [checked] Components: `scripts/sweep.py::write_report` — validated report generation (named code exists and changed)
  - [tested] How it was verified: Ruff passed for `scripts/sweep.py` and `tests/test_experiments.py`. (test gate passed: project test suite (CPU))
  - [claimed] Components: `tests/test_experiments.py::test_gpu_report_uses_completed_metrics_and_rejects_proxy` —... (names no code)
  - [claimed] Key parameters: Seed 42: `configs/sweeps/gpu_baseline.yaml::seeds`; fixes stochastic initialization and... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing `val_metrics.json` and proxy-marked results raise an error in the repo... (`val_metrics.json` not found)
  - [claimed] How it was verified: Inspected all six run directories and confirmed each has `val_metrics.json`, `metrics.c... (no test named)
  - [claimed] Components: `docs/sweep_results.md` — observed results (names no code)
  - [claimed] Key parameters: Batch size 71: `configs/sweeps/gpu_baseline.yaml::overrides`; matches all six completed... (no cited constant with a matching value)
  - full per-claim list (24) in the JSON record

**Consistency**

0 error(s), 4 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: val_metrics.json
- WARN: references something that doesn't exist: metrics.csv
- WARN: references something that doesn't exist: summary.md
- WARN: references something that doesn't exist: config.yaml

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0022-gpu-baseline-sweep.md` gives the commit.
<!-- verified-facts:end -->
