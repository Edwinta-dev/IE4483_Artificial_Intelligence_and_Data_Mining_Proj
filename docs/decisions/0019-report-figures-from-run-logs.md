# 0019: Report figures from run logs
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #19 · Date: 2026-10-19
## In short
Completed run folders can now be turned into per-run loss and accuracy plots and a combined validation-accuracy overlay. The reporting command also writes CSV and Markdown summaries using the saved run config and epoch metrics; proxy dataset runs carry a visible `PROXY` label.
## How it works
1. The user supplies completed run folders and an output folder to the report command. Each run must contain its saved configuration and epoch metrics, as produced by training.
2. For each run, the command reads `metrics.csv` for its epoch values and timing, reads `config.yaml` for the backbone, freeze setting, and learning rate, and uses the saved transform description for the augmentation column. The existing training-config loader handles extra training settings stored with real run configs.
3. It chooses the epoch with the highest validation accuracy and draws train and validation loss and accuracy against epoch. A dashed vertical line marks that selected epoch. The figure is saved in both PNG and PDF formats inside that run folder.
4. It checks the configured dataset root for the existing proxy marker. When present, the per-run figure title and comparison row identify the result as a proxy.
5. The output folder receives an overlay of validation-accuracy curves, also as PNG and PDF, plus `comparison.csv` and `comparison.md`. The tables include configuration, completed epoch count, best validation accuracy, and mean recorded seconds per epoch. These artifacts report existing logs and do not trigger training or evaluation.
## Main design choice
The report reads each run's persisted metrics and config directly. An alternative was to re-run evaluation and rebuild summaries from `val_metrics.json`, but that would require data and model loading and could produce numbers detached from the historical epoch curve or fail when the original dataset is unavailable. Direct log parsing keeps the table and charts traceable to the run artifacts and works on CPU without weights. The best epoch is selected from `val_acc`, matching the trainer's checkpoint-selection rule; using the final epoch would misrepresent early-stopped or non-monotonic runs.
## Traps: why the obvious fix is wrong
- `metrics.csv` rows contain both validation measurements and training time. Best accuracy must be selected by the maximum validation accuracy, while time per epoch is the average of recorded `seconds`; selecting the last row or summing time would answer a different question.
- The run config contains trainer-specific settings beyond the typed config. `scripts/report_figures.py::_summarize_run` uses `src/ie4483/train.py::load_training_config` so valid saved configs with extra train fields remain readable.
- `freeze_backbone` is the stored typed setting and maps to the displayed legacy modes `frozen` and `full`. Missing or unreadable run artifacts should fail visibly rather than produce empty tables.
- Proxy status comes from the dataset marker at the configured data root, the same source used by evaluation. A run name alone is not evidence that a dataset is a proxy.
## Components
- `scripts/report_figures.py::report_figures` — parse runs and write tables
- `scripts/report_figures.py::main` — command line
- `src/ie4483/figures.py::training_curves` — per-run metric chart
- `src/ie4483/figures.py::comparison_figure` — validation overlay
- `src/ie4483/figures.py::_save_report_figure` — PNG/PDF styling and save
- `tests/test_figures.py` — synthetic run report coverage
## Breaks or needs follow-up
None. The report is additive and does not change training, evaluation, or sweep behavior.
## Problem and constraints
Rubric parts (c), (d), and (f) need report-ready charts and comparison values grounded in run logs. The implementation processes existing artifacts only; tests create synthetic run folders and do not use project data, pretrained weights, or a GPU.
## Alternatives considered
- Re-evaluate each checkpoint: requires original data and model construction, and would not provide the recorded per-epoch training curve.
- Add report generation to every experiment runner: would couple training execution to report destinations and omit runs created by the normal training command.
## Trade-offs
The script expects the normal run artifact names and columns. Renaming trainer metrics or removing `transforms.txt` changes report behavior; when transforms are absent, the config augmentation mapping is shown instead. The combined overlay compares validation accuracy only, leaving the full loss and train/validation detail in individual figures.
## Key parameters
- `metrics.csv::val_acc` — selects the marked best epoch and best validation accuracy; a different selection metric would change the report's highlighted point.
- `metrics.csv::seconds` — averaged across recorded rows for time per epoch; missing timing data makes the report fail rather than invent a value.
- `--runs` and `--out` (`scripts/report_figures.py::main`) — explicit input run folders and artifact destination.
## Assumptions
Run folders are complete enough to contain non-empty `metrics.csv` and `config.yaml`. Their metrics use the schema written by `src/ie4483/train.py`, and the configured data root is still available for proxy-marker inspection. Accuracy and loss values are copied from logs without recalculation.
## Edge cases
- Handled: non-consecutive epoch numbers; proxy labels; missing transform descriptions use config augmentation values; output folders are created.
- Not handled: empty run lists, empty metrics, or missing required metric/config files produce a clear exception; malformed rows are not repaired.
## Not verified
No report was run against real experiment data. Proxy recognition was verified using a synthetic marker, not an owner-created proxy training run.
## Changes outside the scope
- `tests/test_figures.py`: required regression coverage for the new report output and marker.
- `README.md`: documents the new command, as required for added commands.
- `docs/decisions/0019-report-figures-from-run-logs.md`: required decision record.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_figures.py` passed 4 tests. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check scripts/report_figures.py src/ie4483/figures.py tests/test_figures.py` passed. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed all 71 tests on CPU in 154.59 seconds after the final parser adjustment. Pytest reported the existing scheduler-order warning in `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`; it did not fail a test.
## To change this
- To change the highlighted epoch rule, change the maximum selection over `val_acc` in `src/ie4483/figures.py::training_curves` and `scripts/report_figures.py::_summarize_run` together; both the marker and table must describe the same epoch.
- To add curves to the combined plot, extend `src/ie4483/figures.py::comparison_figure`; its current comparison is validation accuracy.
- To change CSV columns, update `scripts/report_figures.py::FIELDS` and the Markdown header/row construction together.
## Rollback
Remove `scripts/report_figures.py` and the report-curve helpers in `src/ie4483/figures.py`; revert the README command and remove the report tests and this decision record. Existing training and sweep code does not import these report helpers.


<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:54 UTC for issue #19 attempt 6; compared HEAD..working tree (base 7799a52). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 0 component(s), +92/-0 lines
- `src/ie4483/`: 1 file(s), 3 component(s), +44/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/report_figures.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/report_figures.py`, `src/ie4483/figures.py`, `src/ie4483/experiments.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

5 checked by the tool, 1 tested in this run, 13 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 5 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `scripts/report_figures.py::report_figures` — parse runs and write tables (named code exists and changed)
  - [tested] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_figures.py` pas... (test gate passed: project test suite (CPU))
  - [checked] Components: `scripts/report_figures.py::main` — command line (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::training_curves` — per-run metric chart (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::comparison_figure` — validation overlay (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::_save_report_figure` — PNG/PDF styling and save (named code exists and changed)
  - [claimed] Components: `tests/test_figures.py` — synthetic run report coverage (names no code)
  - [claimed] Key parameters: `metrics.csv::val_acc` — selects the marked best epoch and best validation accuracy; a... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: non-consecutive epoch numbers; proxy labels; missing transform descriptions us... (names no code)
  - [claimed] Key parameters: `metrics.csv::seconds` — averaged across recorded rows for time per epoch; missing timi... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 7 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 4 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: references something that doesn't exist: metrics.csv::val_acc (no such file)
- WARN: references something that doesn't exist: config.yaml
- WARN: references something that doesn't exist: comparison.csv
- WARN: references something that doesn't exist: comparison.md
- WARN: references something that doesn't exist: val_metrics.json
- WARN: references something that doesn't exist: transforms.txt

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0019-report-figures-from-run-logs.md` gives the commit.
<!-- verified-facts:end -->
