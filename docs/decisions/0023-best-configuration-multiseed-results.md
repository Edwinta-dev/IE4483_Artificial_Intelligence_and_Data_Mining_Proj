# 0023: Best configuration multi-seed results
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #23 · Date: 2026-10-27
## In short
The selected ResNet-50 frozen configuration has completed validation runs for seeds 0, 1, and 2 (`runs/seeds/best/summary.csv`). The results page records their mean and spread, seed 1's validation confusion matrix and incorrect examples, and the 500 test-image rows (`submission.csv`) generated from seed 1's best checkpoint.
## How it works
1. The completed baseline sweep identifies the ResNet-50 frozen setting as the configuration to repeat. The owner runs it with the fixed seeds listed in `runs/seeds/best/summary.csv` and retains the run folders locally.
2. Each run's completed validation result supplies one accuracy for the seed table. The summary reports the arithmetic mean and population standard deviation across those three values.
3. The checkpoint with the highest validation accuracy is evaluated again on the labeled validation images. Its confusion matrix and confidence-ranked correct and incorrect image galleries are saved with that run.
4. The same checkpoint predicts labels for the unlabeled test images. The output is sorted by numeric image ID and uses 1 for dog and 0 for cat.
5. The figure command reads the sweep and seed runs listed in `runs/report/issue23/comparison.csv`. Its comparison figure, the selected run's curves and confusion matrix are copied into the report figures folder; the incorrect galleries contain dataset photos, so they stay in `runs/report/issue23/galleries/`; source logs and other artifacts remain in ignored run folders.
## Main design choice
Use the owner's saved runs and the repository's existing evaluation, prediction, and reporting commands. The alternatives were to reevaluate every checkpoint or generate a fresh report pipeline. Reevaluating all checkpoints would spend time without changing which selected checkpoint is needed, and a new pipeline would duplicate existing writers with a higher chance of inconsistent class ordering or metric conventions. Reusing the persisted runs won because it preserves their configuration, seed identity, and checkpoint provenance. A fresh evaluation would be preferable if the saved validation metrics were missing or incompatible; all three runs instead contain completion markers and were independently summarized.
## Traps: why the obvious fix is wrong
- A seed summary row alone is not proof of a finished run; each row was checked against its run's `val_metrics.json`.
- The top checkpoint is seed 1, not the first seed directory. Choosing by folder order would select a lower validation result.
- The confusion matrix is oriented as true class by predicted class, with `[cat, dog]` on both axes. Reversing either axis changes the meaning of the off-diagonal counts.
- Test labels must not affect checkpoint choice. The checkpoint was selected using validation accuracy, then used for test prediction.
- The submission IDs are numeric and dog is label 1. Text sorting or swapping class columns would violate the requested submission contract.
## Components
- `docs/results.md` — result narrative
- `docs/figures/` — copied report figures
- `submission.csv` — final test labels
- `runs/seeds/best/summary.csv` — seed metrics
- `scripts/evaluate.py::main` — selected validation reevaluation
- `scripts/predict.py::predict` — test inference
- `scripts/report_figures.py::report_figures` — sweep and seed charts
## Breaks or needs follow-up
None. The tracked additions are report artifacts and do not change training or evaluation behavior.
## Problem and constraints
Issue #23 needs a three-seed result summary, validation case galleries, report figures, and a test submission. Training was already completed by the owner. The task permits CUDA for evaluation, prohibits further training and downloads, and requires result figures under `docs/` while preserving the original run artifacts under ignored `runs/`.
## Alternatives considered
- Leave the prior record blocked because it claimed the seed folder was missing — rejected after checking the current workspace, where all three completed run folders exist.
- Report only the existing summary file — insufficient because the issue also requires galleries, figures, a narrative, and a root submission file.
- Select the seed by its smallest or largest index — unrelated to model selection; seed 1 has the highest observed validation accuracy.
## Trade-offs
The written results and figures are a snapshot of the owner-run data and hardware. The detailed source run folders are ignored and need to be retained locally for full reproduction; only the report artifacts and submission are tracked.
## Key parameters
- Seeds 0, 1, and 2: required run set in `runs/seeds/best/summary.csv`; adding seeds changes the estimated spread and requires regenerating the table.
- Validation mean 0.995266 and population standard deviation 0.000115: computed from `runs/seeds/best/summary.csv`; more precision can be shown by changing formatting in `docs/results.md`.
- Confusion counts 2484, 15, 8, and 2492: `runs/seeds/best/seed_1/*/issue23_eval.json`; a different checkpoint or dataset changes the matrix.
- Test row count 500: `submission.csv` after its header; changing the test set changes the expected row count.
## Assumptions
The completed run folders correspond to the cleaned Dogs vs Cats data and the selected baseline configuration documented in `docs/sweep_results.md`. Their saved configurations and results show the ResNet-50 frozen setup and non-proxy data. If those source runs or the data handoff are replaced, regenerate the results page, copied figures, and submission together.
## Edge cases
- Handled: selected seed is determined from observed validation accuracy; all three seed completion files are present; the submission header and numeric ordering are checked.
- Not handled: source run artifacts are not committed, so a later reader without the owner's local `runs/` folder cannot regenerate the same figures or probabilities.
## Not verified
The visual galleries were inspected for repeated qualitative patterns, but no formal causal analysis or annotation audit was performed. The test labels are unavailable, so test accuracy is not measured.
## Changes outside the scope
None. Tracked outputs and documentation are within `docs/**`, `README.md`, and `submission.csv`.
## How it was verified
Inspected completion markers for all three seed runs and six sweep runs. Re-evaluated seed 1 on validation, predicted the test split, and generated the sweep-plus-seed figure report. Checked the submission header, 500 numeric-sorted unique test IDs, the three-row seed summary, confusion matrix values, all five figure references, and all 17 required decision-record headings. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_evaluate.py tests/test_submission.py tests/test_figures.py tests/test_experiments.py` passed 17 tests. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 79 tests in 113.26 seconds; one existing scheduler-order warning appeared in `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`.
## To change this
- To select a different configuration, update `docs/sweep_results.md` first, rerun all seeds with that configuration, and regenerate every report artifact; cost: new owner training runs.
- To include more examples in each gallery, adjust `src/ie4483/evaluate.py::_GALLERY_LIMIT` and reevaluate; cost: larger figures and a new decision in a future issue.
- To use another seed as the submission model, select it by validation accuracy, run prediction from its saved checkpoint, and update the described matrix and figures together.
## Rollback
Remove `docs/results.md`, `docs/figures/`, and `submission.csv` to roll back this issue's tracked outputs, and revert the README section and this record. The source training and sweep runs are ignored artifacts and are not modified by rollback.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-26 16:22 UTC for issue #23 attempt 1; compared HEAD..working tree (base 6b17289). Records are claims to check against the code, not proof.

**Change summary**

- `(root)/`: 1 file(s), 0 component(s), +501/-0 lines
- plus 7 doc file(s)
- new files: `submission.csv`

**Removed symbols**

None.

**Scope**

Declared: `configs/**`, `docs/**`, `README.md`, `submission.csv`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), scope check, dangling reference check

**Evidence**

1 checked by the tool, 1 tested in this run, 20 claimed only (of 22 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 1 checked, 0 tested, 6 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `submission.csv` — final test labels (named code exists and changed)
  - [tested] How it was verified: Inspected completion markers for all three seed runs and six sweep runs. Re-evaluated s... (test gate passed: project test suite (CPU))
  - [claimed] Components: `docs/results.md` — result narrative (names no code)
  - [claimed] Key parameters: Seeds 0, 1, and 2: required run set in `runs/seeds/best/summary.csv`; adding seeds chan... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: selected seed is determined from observed validation accuracy; all three seed... (names no code)
  - [claimed] Components: `docs/figures/` — copied report figures (names no code)
  - [claimed] Key parameters: Validation mean 0.995266 and population standard deviation 0.000115: computed from `run... (no cited constant with a matching value)
  - [claimed] Components: `runs/seeds/best/summary.csv` — seed metrics (`runs/seeds/best/summary.csv` did not change)
  - [claimed] Key parameters: Confusion counts 2484, 15, 8, and 2492: `runs/seeds/best/seed_1/*/issue23_eval.json`; a... (no cited constant with a matching value)
  - [claimed] Components: `scripts/evaluate.py::main` — selected validation reevaluation (`scripts/evaluate.py::main` exists but did not change)
  - full per-claim list (22) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: val_metrics.json

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0023-best-configuration-multiseed-results.md` gives the commit.
<!-- verified-facts:end -->
