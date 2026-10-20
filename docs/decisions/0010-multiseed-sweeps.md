# 0010: Multi-seed runner and parameter sweep
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #10 · Date: 2026-10-19
## In short
The project now repeats training and validation across explicit random seeds and summarizes the resulting accuracies. A YAML grid command runs combinations across seeds, reports sorted cell averages and sample standard deviations, and reuses completed seed runs.
## How it works
1. Choose one training configuration and provide one or more seed values, or choose a sweep file that lists shared configuration overrides, settings to compare, and the seeds for each comparison.
2. For each requested seed, the program builds a training configuration with that seed, trains a run, and evaluates its selected checkpoint on the labeled validation images. Each run retains the usual training and evaluation artifacts.
3. The seed command writes one accuracy row per run and adds the overall mean and standard deviation to CSV and Markdown summaries.
4. The sweep command makes one output folder per parameter combination and one subfolder per seed. It checks for the evaluation metrics file before starting a seed; if found, its recorded accuracy is reused. It then writes a sorted table with one row per parameter combination and a bar chart whose error bars show variation across seeds.
5. All generated run folders, summaries, and charts stay under the chosen output folder, which defaults beneath the ignored runs directory.
## Main design choice
The runner calls the existing training and validation APIs in the same Python process. A subprocess-based runner was a real alternative: it would isolate each run and could recover more cleanly from a crashed worker, but would need to duplicate command argument handling and parse output artifacts. The direct API keeps the saved run configuration and validation metrics authoritative and lets tests inspect exact values. A single seed directory per run lets the existing `val_metrics.json` artifact serve as the resume marker; a separate sweep database would add state that could drift from the actual run.
## Traps: why the obvious fix is wrong
- A training metrics row is not evidence that evaluation finished. Resume checks `val_metrics.json`, emitted only after validation evaluation, so partial runs are retrained.
- The sweep treats each parameter combination and seed as its own run location. Reusing one folder for an entire cell could overwrite checkpoints and make it impossible to trace a reported accuracy to its configuration.
- Cell paths include a digest of the original candidate values (`src/ie4483/experiments.py::_cell_name`); relying only on sanitized display text could make distinct values map to the same folder and cause an incorrect resume.
- Standard deviation is the sample standard deviation when there are at least two seeds (`src/ie4483/experiments.py::_write_summary`). For one seed the displayed spread is zero; it does not imply measured seed stability.
- The strict configuration loader only accepts currently modeled keys. The freeze comparison uses `model.freeze_backbone`, and the shipped architecture comparison uses the supported `model.architecture`; adding an unmodeled key such as `model.freeze_mode` would fail config validation.
- The data-root sweep values are resolved from the process working directory by the existing data-root rules. Run the command from the repository root or provide paths meaningful from the current directory.
- Shared sweep overrides are applied before cell values. Keeping a shared key out of the grid ensures every comparison uses the same setting.
## Components
- `src/ie4483/experiments.py::run_seeds` — fixed-seed runner
- `src/ie4483/experiments.py::sweep` — grid, resume, ordering, chart
- `src/ie4483/experiments.py::_write_summary` — CSV and Markdown
- `scripts/run_seeds.py::main` — seed CLI
- `scripts/sweep.py::main` — sweep CLI
- `configs/sweeps/` — four example grids
- `tests/test_experiments.py` — generated-data integration and summaries
## Breaks or needs follow-up
None known. The supplied backbone/freeze grid compares `freeze_backbone` true and false; it does not expose the model factory's `last_block` mode because that key is not accepted by the strict typed config loader.
## Problem and constraints
Rubric parts (b), (c), and (f) need repeatable comparisons rather than a single score. The work uses the existing trainer and evaluator, keeps validation as the model-selection/reporting split, and tests with small generated images, no pretrained weights, and CPU execution. It does not train on the project dataset or fetch any weights.
## Alternatives considered
- Put multi-seed logic only in a shell loop — rejected because it would not produce standardized per-seed summaries or resumable cells.
- Resume based on the presence of a checkpoint — rejected because a checkpoint can exist before validation artifacts are complete.
- Save only aggregate values — rejected because per-seed measurements are needed to inspect the mean and spread and trace results to their run folders.
## Trade-offs
Every seed remains a complete training run with its checkpoint and evaluation figures, so sweeps consume disk space and can take substantial time. Resuming relies on the run folder remaining intact and its metrics file matching its configuration. A metrics file manually copied into another seed folder could be mistaken for a completed run; the tool does not cryptographically validate artifact provenance.
## Key parameters
- Seed values come from `--seeds` or each sweep YAML's `seeds` list; changing them changes the stochastic training repetitions and the reported spread.
- Sweep dimensions and candidate values come from each file's `parameters` mapping; adding values multiplies the number of combinations and training runs.
- The sweep output default is `runs/sweeps/<sweep-name>` (`scripts/sweep.py::main`); `--output` selects another artifact location.
- Summary spread uses sample standard deviation (`statistics.stdev` in `src/ie4483/experiments.py::_write_summary`); the one-sample display is zero because sample spread is undefined for one value.
- Shipped grids use seeds `[0, 1, 2]` (`configs/sweeps/*.yaml`); changing this list changes repetition count and runtime.
- `train.batch_size=4` is a shared override in `configs/sweeps/backbone_freeze.yaml` to follow the project guidance for fine tuning; raising it increases memory use.
## Assumptions
Training and evaluation continue to write complete run artifacts as defined by the existing pipeline. The run folder and its metrics remain available between invocations. Sweep keys and values pass through the normal config validation and override path; unsupported values fail with that loader's error.
## Edge cases
- Handled: no seeds, malformed/empty parameter lists, and missing seed lists raise a clear `ValueError`; a one-seed cell has zero displayed standard deviation; completed seed metrics are reused independently when another seed in the cell is incomplete.
- Not handled: resuming after a partially written or manually altered `val_metrics.json` is not validated beyond reading its `accuracy` field; malformed content raises an error.
## Not verified
GPU behavior, CUDA memory use, interrupted training recovery, and experiments against the real dataset were not measured. Sorting was exercised by code inspection and the integration table; the tiny generated data did not establish meaningful model rankings.
## Changes outside the scope
- `README.md`: documents both added commands and the sweep configuration format, as required for new commands/config keys.
- `tests/test_experiments.py`: adds generated-data integration coverage and seed-summary coverage as required.
- `docs/decisions/0010-multiseed-sweeps.md`: required decision record.
No production source outside the declared Scope was changed.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_experiments.py` passed two tests in 29.95 seconds. The integration test trained and evaluated four CPU runs on generated images, matched reported values to run metrics, checked per-cell averages and standard deviations, the chart, resume skipping, folder-name collision resistance, and shared overrides in saved configs. The summary test checked per-seed rows and aggregate mean/sample standard deviation. Ruff passed for the changed Python files. The full CPU suite passed 43 tests in 73.90 seconds.
## To change this
- To compare more seeds, change the `seeds` list in a sweep YAML or pass more values to `scripts/run_seeds.py::main`; runtime and saved run count increase.
- To compare another setting, add its validated dotted key and candidate values under `parameters`; each added grid value multiplies cell count.
- To change ties or sorting direction, edit the `sorted` ordering in `src/ie4483/experiments.py::sweep`; the current order favors larger mean validation accuracy and keeps input grid order for ties.
- To use a different spread definition, update `statistics.stdev` and the one-seed behavior consistently in the summary and chart paths.
## Rollback
Remove `src/ie4483/experiments.py`, `scripts/run_seeds.py`, `scripts/sweep.py`, `configs/sweeps/`, and `tests/test_experiments.py`; remove the README usage section and this record. Existing train and evaluate commands do not import the experiment runner.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 03:00 UTC for issue #10 attempt 10; compared HEAD..working tree (base ab89260). Records are claims to check against the code, not proof.

**Change summary**

- `configs/sweeps/`: 4 file(s), 0 component(s), +18/-0 lines
- `scripts/`: 2 file(s), 0 component(s), +46/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +166/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `configs/sweeps/augmentation.yaml`, `configs/sweeps/backbone_freeze.yaml`, `configs/sweeps/learning_rate.yaml`, `configs/sweeps/raw_vs_cleaned.yaml`, `scripts/run_seeds.py`, `scripts/sweep.py`, `src/ie4483/experiments.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/experiments.py`, `scripts/run_seeds.py`, `scripts/sweep.py`, `configs/sweeps/**`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

6 checked by the tool, 0 tested in this run, 20 claimed only (of 26 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 6 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 6 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 7 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/experiments.py::run_seeds` — fixed-seed runner (named code exists and changed)
  - [checked] Components: `src/ie4483/experiments.py::sweep` — grid, resume, ordering, chart (named code exists and changed)
  - [checked] Components: `src/ie4483/experiments.py::_write_summary` — CSV and Markdown (named code exists and changed)
  - [checked] Components: `scripts/run_seeds.py::main` — seed CLI (named code exists and changed)
  - [checked] Components: `scripts/sweep.py::main` — sweep CLI (named code exists and changed)
  - [checked] Components: `configs/sweeps/` — four example grids (named code exists and changed)
  - [claimed] Components: `tests/test_experiments.py` — generated-data integration and summaries (names no code)
  - [claimed] Key parameters: Seed values come from `--seeds` or each sweep YAML's `seeds` list; changing them change... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: no seeds, malformed/empty parameter lists, and missing seed lists raise a clea... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_experiments.py`... (no test named)
  - full per-claim list (26) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: val_metrics.json

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0010-multiseed-sweeps.md` gives the commit.
<!-- verified-facts:end -->
