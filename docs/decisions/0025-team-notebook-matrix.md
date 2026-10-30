# 0025: Team notebook matrix from finished runs
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #25 · Date: 2026-10-27

## In short

The repository now has a text-only exporter for the completed seed-42 matrix, per-epoch curves, confusion counts, seed runs, misclassifications, data counts, adjustment examples, and test submission statistics. A cleared-output notebook reads those summaries to create the report charts and tables; its locally rendered HTML and temporary execution files stay under ignored `runs/report/`.

## How it works

1. The exporter checks the matrix completion marker, locates the saved run folders for each backbone, freeze mode, and pipeline, and requires each run's validation completion file. It reads existing logs and predictions; it does not train, evaluate a checkpoint, or download anything.
2. For each matrix cell, the report takes the saved validation result, counts logged epochs, averages their recorded durations, and counts validation true positives, false positives, true negatives, and false negatives. It chooses the highest-accuracy cell for the error list.
3. The exporter gathers the three saved seed groups, the selected seed's submission and prediction files, local split counts, and one saved adjustment example of each kind. It writes CSV and JSON summaries containing text and numbers only.
4. The notebook loads those summary files and draws the comparisons, curves, seed spread, confusion matrix, and submission histogram. If local `data/` images exist, it displays sample images and applies the saved adjustment parameters in memory for previews; otherwise those previews are skipped. Notebook outputs remain cleared, and the executed HTML is written under ignored `runs/report/`.

## Main design choice

The report consumes completed run artifacts directly. Re-running evaluation could refresh metrics, but would require checkpoint and dataset access and could make the comparison describe a different evaluation environment. Training or using a new scoring path was also possible, but would spend time and introduce new result provenance. Reading the saved validation files won because this issue asks for a report from finished runs and the existing artifacts preserve the selected model result, epoch history, prediction labels, and source run path. If a saved validation result or prediction file is missing, the exporter stops instead of manufacturing a value.

## Traps: why the obvious fix is wrong

- Do not treat an epoch log as proof of completion: the exporter requires `val_metrics.json` before using a run.
- Do not silently report a partial grid: the exporter checks all backbone and freeze combinations in every pipeline and fails on a missing cell.
- The class-1 prediction is dog, so the confusion counts and `p_dog` error values follow the project order `[cat, dog]` (`configs/base.yaml::data.classes`).
- Submission labels have no confidence values. Label counts come from `submission.csv`; confidence comes from its paired saved `test_predictions.csv`, and is not presented as test accuracy.
- Adjustment previews are reconstructed in memory from saved parameters and source images under `data/`; the summary contains no image bytes.

## Components

- `scripts/export_report_summary.py::export_summary` — assemble report tables
- `scripts/export_report_summary.py::_run_dir` — completion guard
- `scripts/export_report_summary.py::main` — exporter CLI
- `notebooks/cats_vs_dogs_report.ipynb::notebook` — report views
- `tests/test_report_summary.py::test_export_summary_writes_matrix_curves_seeds_and_error_counts` — output contract
- `tests/test_report_summary.py::test_export_summary_rejects_run_without_completion_marker` — unfinished run guard
- `tests/test_report_summary.py::test_export_summary_rejects_missing_matrix_cell` — complete grid guard

## Breaks or needs follow-up

None. The existing training and evaluation commands are unchanged.

## Problem and constraints

The team needs a notebook covering the backbone, freeze, and pipeline matrix plus data, cost, seed, error, and submission views. Runs are already complete; the report must not train or make the dataset images public. The notebook reads summary tables and uses local data images only for explicitly optional previews. The expected matrix is defined by `scripts/export_report_summary.py::BACKBONES`, `scripts/export_report_summary.py::FREEZES`, and `scripts/export_report_summary.py::PIPELINES`.

## Alternatives considered

- Reevaluate every checkpoint — not chosen because it adds model and data loading and can replace the historical saved validation results.
- Generate the whole report from a Python plotting script — not chosen because the requested team deliverable is an editable notebook with explanatory reading beside each view.

## Trade-offs

Saved summaries are quick to regenerate and preserve run provenance, but cannot repair an incomplete or inconsistent source run. Seed spread is available only for the saved ResNet-50 frozen groups; other matrix comparisons remain single-run descriptions. The HTML is a local share artifact and is not tracked.

## Key parameters

- Seed-42 matrix identity: `runs/sweeps/gpu_baseline`, `runs/sweeps/gpu_aug`, and `runs/sweeps/gpu_mixed`; the exporter expects two backbones, three freeze modes, and three pipelines (`scripts/export_report_summary.py::BACKBONES`, `scripts/export_report_summary.py::FREEZES`, `scripts/export_report_summary.py::PIPELINES`).
- Seed spread groups: `runs/seeds/best`, `runs/seeds/aug`, and `runs/seeds/mixed`; each contains the saved seed folders read by `scripts/export_report_summary.py::export_summary`.
- Confidence histogram intervals are defined by the `bins` list inside `scripts/export_report_summary.py::export_summary`; changing the edges changes the displayed distribution, not any prediction.
- The six removed duplicates and split totals are stored in `runs/report/summary/data_counts.json`; counts are read from the local data folders and the duplicate count follows the issue's handoff fact.

## Assumptions

Each required matrix cell and seed folder contains exactly one run subfolder. Saved prediction paths refer to the local data layout when image previews are attempted. If either assumption fails, summary generation errors or the corresponding optional images are skipped.

## Edge cases

- Handled: absent completion marker, missing matrix cells, missing validation completion marker, missing validation predictions, proxy-marked results, and absent local image folders.
- Not handled: stale but present run artifacts or mismatched data provenance; the report shows their saved values and paths, so the owner must replace or audit the source artifacts before regenerating.

## Not verified

No independent comparison against a fresh checkpoint evaluation was performed. The report does not estimate variability for the single-seed matrix cells.

## Changes outside the scope

None.

## How it was verified

`C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/export_report_summary.py` generated the summaries from the local completed runs. `tests/test_report_summary.py` passed all three exporter tests; Ruff passed for the exporter and its tests. `jupyter nbconvert --execute --to html --output cats_vs_dogs_report.html --output-dir runs/report notebooks/cats_vs_dogs_report.ipynb` executed successfully and wrote the ignored HTML. The committed notebook has 21 cells and all code-cell outputs cleared. The full CPU suite passed: 84 tests in 111.77 seconds; the existing learning-rate scheduler-order warning appeared in `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`.

## To change this

- To add a backbone or freeze mode, update `scripts/export_report_summary.py::BACKBONES` or `scripts/export_report_summary.py::FREEZES`, then update notebook ordering and rerun the exporter and notebook; cost: more source runs and chart/table cells.
- To change the confidence bins, edit the `bins` list in `scripts/export_report_summary.py::export_summary`; cost: histogram counts will no longer match earlier report versions.
- To select a different error-analysis run, change the validation-accuracy selection in `scripts/export_report_summary.py::export_summary`; cost: confusion and misclassification views will describe a different cell.

## Rollback

Remove the exporter, notebook, exporter tests, and `runs/report/summary/` unignore rules, then remove the README regeneration section and this decision record. The training and evaluation pipeline has no imports from these report components.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-27 08:17 UTC for issue #25 attempt 1; compared HEAD..working tree (base ecf5155). Records are claims to check against the code, not proof.

**Change summary**

- `(root)/`: 1 file(s), 0 component(s), +5/-1 lines
- `notebooks/`: 1 file(s), 0 component(s), +314/-0 lines
- `runs/report/`: 9 file(s), 0 component(s), +259/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +162/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `notebooks/cats_vs_dogs_report.ipynb`, `runs/report/summary/adjustment_examples.csv`, `runs/report/summary/adjustments.csv`, `runs/report/summary/confusion.csv`, `runs/report/summary/curves.csv`, `runs/report/summary/data_counts.json`, `runs/report/summary/matrix.csv`, `runs/report/summary/misclassified.csv`, `runs/report/summary/seeds.csv`, `runs/report/summary/submission_stats.json`, `scripts/export_report_summary.py`

**Removed symbols**

None.

**Scope**

Declared: `notebooks/cats_vs_dogs_report.ipynb`, `scripts/export_report_summary.py`, `runs/report/summary/**`, `.gitignore`, `README.md`, `docs/**`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 1 tested in this run, 16 claimed only (of 21 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 3 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `scripts/export_report_summary.py::export_summary` — assemble report tables (named code exists and changed)
  - [tested] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/export_report_summary.py` gener... (test gate passed: project test suite (CPU))
  - [checked] Components: `scripts/export_report_summary.py::_run_dir` — completion guard (named code exists and changed)
  - [checked] Components: `scripts/export_report_summary.py::main` — exporter CLI (named code exists and changed)
  - [checked] Components: `notebooks/cats_vs_dogs_report.ipynb::notebook` — report views (named code exists and changed)
  - [claimed] Components: `tests/test_report_summary.py::test_export_summary_writes_matrix_curves_seeds_and_error... (names no code)
  - [claimed] Key parameters: Seed-42 matrix identity: `runs/sweeps/gpu_baseline`, `runs/sweeps/gpu_aug`, and `runs/s... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: absent completion marker, missing matrix cells, missing validation completion... (names no code)
  - [claimed] Components: `tests/test_report_summary.py::test_export_summary_rejects_run_without_completion_marke... (names no code)
  - [claimed] Key parameters: Seed spread groups: `runs/seeds/best`, `runs/seeds/aug`, and `runs/seeds/mixed`; each c... (no cited constant with a matching value)
  - full per-claim list (21) in the JSON record

**Consistency**

0 error(s), 3 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: val_metrics.json
- WARN: references something that doesn't exist: test_predictions.csv
- WARN: .gitignore (modified) changed but not mentioned in the record

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0025-team-notebook-matrix.md` gives the commit.
<!-- verified-facts:end -->
