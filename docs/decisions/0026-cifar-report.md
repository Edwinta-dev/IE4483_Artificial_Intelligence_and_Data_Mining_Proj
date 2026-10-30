# 0026: CIFAR-10 balanced baselines and cat imbalance report
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #26 · Date: 2026-10-28

## In short
All 14 completed balanced and imbalanced CIFAR-10 checkpoints (`scripts/cifar_report.py::EXPECTED_RUNS`) have test-split metrics in their run folders. A report now summarizes those frozen evaluations in a table, two confusion views, a seed-spread chart and a missed-cat breakdown, with a short results narrative.

## How it works
1. The evaluator loads each existing `best.pt` and measures it on the official CIFAR-10 test batch. It writes `test_metrics.json` and the split-specific confusion image into the corresponding run folder; no training is involved.
2. The report generator searches the expected baseline and remedy run folders. It reads each run's saved configuration and test metrics, checks for a full 10,000-example result (`scripts/cifar_report.py::TEST_COUNT`) and a ten-class confusion matrix, and requires all 14 runs to be present (`scripts/cifar_report.py::EXPECTED_RUNS`).
3. It writes one CSV row per run with accuracy, macro-F1, macro recall, cat recall and cat precision. Its JSON keeps those values, the confusion matrices and per-fix mean missed-cat destinations as numeric data.
4. It draws row-normalized matrices for balanced ResNet-50 and no-fix imbalanced ResNet-18. A second chart compares mean cat recall and accuracy for the four remedies, with standard-deviation error bars and balanced ResNet-18 reference lines.
5. The Markdown report combines the table and figures with the observed cat-recall drop, the strongest fix, its dog-recall change and a comparison smaller than the seed spread. It explicitly identifies the balanced runs as single-seed references.

## Main design choice
The report uses the project's existing evaluator for test inference and keeps its report generator as a consumer of the resulting `test_metrics.json` files. Reimplementing checkpoint loading and metrics inside the report script was a real alternative, but would duplicate the evaluator and risk metric differences. Reading validation metrics or predictions alone would be cheaper, but would fail the issue's official-test requirement. The chosen path keeps test-set evaluation explicit and rerunnable, while generation of tables and figures does not need the model weights or dataset after metrics are saved.

## Traps: why the obvious fix is wrong
- A completed training marker such as `val_metrics.json` does not provide a test score. The test evaluation must run for every checkpoint before report generation.
- Do not use the test values to select checkpoints or remedy settings. The saved `best.pt` was selected on validation; these results describe that already-selected checkpoint.
- The cat is CIFAR label 3, but named `per_class.cat` avoids relying on a potentially changed numeric position for report metrics. The confusion row follows official class order from `configs/cifar10.yaml::data.classes`.
- Do not pool the three seed values as if they were examples. The chart summarizes run-level means and sample standard deviations; the two baseline runs have no seed spread estimate.
- A confusion matrix should be normalized by true-class row to show where cats go conditionally on being cats. The counts table separately reports the average raw cat-row destinations.

## Components
- `scripts/cifar_report.py::generate_report` — tables and figures
- `scripts/cifar_report.py::_runs` — expected run and metric checks
- `scripts/cifar_report.py::main` — report CLI
- `tests/test_cifar_report.py::test_cifar_report_reads_tiny_fake_runs_and_writes_outputs` — fake artifact test
- `docs/cifar_results.md` — results narrative and tables
- `runs/report/cifar/cifar_test_results.json` — numeric results
- `docs/figures/cifar_*.png` — report charts

## Breaks or needs follow-up
None. This adds a reporting workflow and does not change training, checkpoint selection or evaluation defaults.

## Problem and constraints
The report needs actual frozen test metrics for two balanced models and twelve imbalanced runs, with the official test split reserved for final measurement. Training is prohibited in this issue. The numeric files belong in the one unignored `runs/report/cifar/` folder; figures and the readable report belong under `docs/`.

## Alternatives considered
- Infer from each sweep's summary CSV — rejected because those summaries are validation results, not official test evaluations.
- Duplicate checkpoint inference in the report script — rejected to keep metric calculation centralized in `src/ie4483/evaluate.py::evaluate`.
- Put narrative and plots in the numeric output directory — rejected because the rubric calls for a readable report and figures under `docs/`.

## Trade-offs
The run artifacts retain an additional full test metric JSON and confusion image per evaluated checkpoint. That costs disk space but preserves provenance and means the report can be regenerated without inference. The chart uses sample standard deviation from only three seeds, which conveys observed spread but is not a confidence interval.

## Key parameters
- `metrics.count == 10000` (`scripts/cifar_report.py::_runs`) — accepts complete official CIFAR test evaluations; a different count is rejected rather than silently reported.
- Three imbalanced seeds 0, 1 and 2 (`configs/sweeps/cifar_imb_none.yaml::seeds`, also the other three remedy sweep configs) — defines the shown mean and sample spread; adding seeds makes variability estimates more stable but requires completed runs.
- Cat class index 3 (`configs/cifar10.yaml::data.classes`) — identifies the cat confusion row; a class-order change must update the meaning of that row.
- Figure resolution 160 dpi (`scripts/cifar_report.py::FIGURE_DPI`) — keeps report figures readable at modest file size; raising it increases image size.

## Assumptions
The finished run folders correspond to the supplied sweep configurations and their `best.pt` files are the intended validation-selected checkpoints. Test metrics written by `scripts/evaluate.py --split test` reflect the official local CIFAR test batch. If the data or checkpoint is replaced in place, the report will show the newer metrics under the same run identity.

## Edge cases
- Handled: missing run results, malformed metrics, incomplete test counts, missing cat metrics, and confusion matrices with the wrong class count raise errors rather than producing a partial report.
- Not handled: the exporter does not hash test data or checkpoints, and does not prove that a hand-created `test_metrics.json` came from the official test evaluator.

## Not verified
The code does not independently compare model predictions with upstream CIFAR labels. The 14 reported results were produced by the project's evaluator using the locally available official test batch; no external provenance audit was performed. No significance test was calculated.

## Changes outside the scope
None.

## How it was verified
All 14 checkpoints (`scripts/cifar_report.py::EXPECTED_RUNS`) were evaluated with `scripts/evaluate.py --split test`; each corresponding run already had its `val_metrics.json` completion marker. `scripts/cifar_report.py` generated the report artifacts from the saved test metrics. `tests/test_cifar_report.py` passed (1 test); the final full CPU suite passed (85 tests in 186.03 seconds), with the existing scheduler-order warning in `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`. A preceding full-suite run also passed in 172.98 seconds. Ruff passed on the report script and test.

## To change this
To add a remedy, extend `scripts/cifar_report.py::FIXES` and its expected seed count in `scripts/cifar_report.py::_runs`, then add its confusion and summary interpretation. To change displayed classes, update `scripts/cifar_report.py::CLASSES` together with `configs/cifar10.yaml::data.classes`. To change report destinations, update the CLI defaults in `scripts/cifar_report.py::main` and the corresponding `.gitignore` allowlist.

## Rollback
Remove `scripts/cifar_report.py`, `tests/test_cifar_report.py`, `docs/cifar_results.md`, the three generated figures and the `runs/report/cifar/` allowlist/output. Restore `.gitignore`'s prior report exceptions and remove the README section. Existing evaluator test artifacts can remain in ignored run folders; no training or evaluation component imports this report code.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-28 08:21 UTC for issue #26 attempt 1; compared HEAD..working tree (base 330e4d7). Records are claims to check against the code, not proof.

**Change summary**

- `(root)/`: 1 file(s), 0 component(s), +2/-0 lines
- `runs/report/`: 2 file(s), 0 component(s), +507/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +177/-0 lines
- plus 1 test file(s), 5 doc file(s)
- new files: `runs/report/cifar/cifar_test_results.csv`, `runs/report/cifar/cifar_test_results.json`, `scripts/cifar_report.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/cifar_report.py`, `docs/**`, `runs/report/cifar/**`, `.gitignore`, `README.md`, `tests/test_cifar_report.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

5 checked by the tool, 1 tested in this run, 16 claimed only (of 22 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 3 claimed
- Key parameters: 1 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `scripts/cifar_report.py::generate_report` — tables and figures (named code exists and changed)
  - [checked] Key parameters: Figure resolution 160 dpi (`scripts/cifar_report.py::FIGURE_DPI`) — keeps report figure... (value matches the code: FIGURE_DPI = 160)
  - [tested] How it was verified: All 14 checkpoints (`scripts/cifar_report.py::EXPECTED_RUNS`) were evaluated with `scri... (test gate passed: project test suite (CPU))
  - [checked] Components: `scripts/cifar_report.py::_runs` — expected run and metric checks (named code exists and changed)
  - [checked] Components: `scripts/cifar_report.py::main` — report CLI (named code exists and changed)
  - [checked] Components: `runs/report/cifar/cifar_test_results.json` — numeric results (named code exists and changed)
  - [claimed] Components: `tests/test_cifar_report.py::test_cifar_report_reads_tiny_fake_runs_and_writes_outputs`... (names no code)
  - [claimed] Key parameters: `metrics.count == 10000` (`scripts/cifar_report.py::_runs`) — accepts complete official... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing run results, malformed metrics, incomplete test counts, missing cat me... (names no code)
  - [claimed] Components: `docs/cifar_results.md` — results narrative and tables (names no code)
  - full per-claim list (22) in the JSON record

**Consistency**

0 error(s), 3 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 4 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: references something that doesn't exist: test_metrics.json
- WARN: references something that doesn't exist: val_metrics.json

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0026-cifar-report.md` gives the commit.
<!-- verified-facts:end -->
