# 0009: Test-set prediction and submission writer
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #9 · Date: 2026-10-19
## In short
A run can now predict its unlabeled test images and write both the required submission file and a probability file for test case studies. Output is ordered by numeric image ID, and the run's class mapping is checked before files are written.
## How it works
1. The operator points the prediction command at a completed run folder and can optionally request horizontal-flip test-time augmentation.
2. The command loads the resolved configuration saved with that run, checks that its class order puts dog at index one, and uses the configured data root to find numeric image IDs under `test/`.
3. It rebuilds the same model architecture, loads the model state from the run's best checkpoint, and applies the run's evaluation image transform. Prediction runs on CUDA when available and otherwise on CPU.
4. Without augmentation, the model's class probabilities are used directly. With flip augmentation, probabilities for each original image and its horizontal mirror are averaged.
5. Before writing, the output helper checks that all discovered test images have predictions, IDs are unique, labels are binary, and the class mapping has the required dog index. It sorts records by numeric ID and writes the exact two-column submission contract plus a second CSV containing cat and dog probabilities.
## Main design choice
Prediction reconstructs the model and preprocessing from the run's saved config and checkpoint, following `src/ie4483/evaluate.py::evaluate`. A separate caller-supplied config would be shorter for ad hoc use, but it could silently change image transforms or architecture from the trained run. Keeping the run folder authoritative makes output repeatable. Flip TTA averages probabilities from two views; averaging logits was also viable, but probability averaging gives the saved class likelihoods a direct interpretation and works with the case-study CSV.
## Traps: why the obvious fix is wrong
- Lexicographic sorting puts ID 10 before ID 2; IDs must be sorted as integers for the submission contract.
- Probability column one is dog only when the saved class list puts dog at index one. The writer checks this instead of silently emitting labels with reversed meaning.
- The training checkpoint is an artifact mapping; model weights live under its `model` entry, as in validation evaluation.
- Unreadable test images are an error for prediction rather than a silent reduction of output rows, because the submission must have one row for every test image.
## Components
- `src/ie4483/submission.py::validate_predictions` — output invariants
- `src/ie4483/submission.py::write_prediction_csvs` — CSV artifacts
- `scripts/predict.py::predict` — run reconstruction and inference
- `scripts/predict.py::main` — CLI
- `tests/test_submission.py` — generated test images and output checks
## Breaks or needs follow-up
None. The command adds artifacts inside the run folder and does not change existing training or evaluation behavior.
## Problem and constraints
Rubric (d) requires an ID and binary label for each test image; rubric (e) needs the associated probabilities to inspect test cases. This is pipeline groundwork only. Tests create tiny local images and use a small model stub; no real data, downloads, cached pretrained weights, or GPU are required.
## Alternatives considered
- Extend validation evaluation to also predict test images — not chosen because submission generation is a separate rubric deliverable and should not change the validation result contract.
- Read the sample submission at runtime — not chosen because its owner copy is outside the repository and the required header and ordering are already explicit in this repository's contract.
- Average logits for flip TTA — not chosen because the output artifact records probabilities and averaging probabilities directly keeps each view's class distribution interpretable.
## Trade-offs
The command holds all IDs and two class probabilities in memory until validation and sorting finish. This is small for the project's fixed test set and permits validation before creating either CSV; a much larger test collection could write a temporary sorted stream instead. Flip TTA performs a second forward pass and therefore takes longer when enabled.
## Key parameters
- Batch size: `config.eval.batch_size` loaded by `scripts/predict.py::predict`; increasing it reduces loader iterations but raises memory use, while decreasing it lowers memory use.
- Worker count: saved training setting `extra["num_workers"]` loaded by `scripts/predict.py::predict`; more workers may improve image loading throughput but add CPU and process overhead.
- Flip TTA: `--tta flip` in `scripts/predict.py::main`; enabling it averages original and mirrored probabilities and roughly doubles image forward passes.
- Class order: `config.data.classes` and the index-one dog requirement in `src/ie4483/submission.py::validate_predictions`; a different dog index is rejected to preserve the required output label meaning.
## Assumptions
The run folder has compatible `config.yaml` and `best.pt` artifacts, and the configured data root contains numeric test-image filenames. The existing test dataset helper ignores non-image files and nonnumeric names; those are outside the specified numeric-ID test layout.
## Edge cases
- Handled: numeric sorting, duplicate IDs, empty test sets (valid header-only files), wrong class order, mismatched row counts, and unreadable images.
- Not handled: files with nonnumeric stems are ignored by the existing dataset helper; a future broader test layout would need to define IDs for them. Incompatible checkpoints surface the underlying load error.
## Not verified
No real test dataset, cached pretrained weights, or CUDA device was used. The horizontal-flip path is implemented, the generated integration test exercises both default prediction and flip TTA.
## Changes outside the scope
- `tests/test_submission.py`: required generated-data acceptance and failure checks.
- `README.md`: documents the added command as required by project instructions.
- `docs/decisions/0009-test-set-prediction.md`: required decision record.
No other production files outside `src/ie4483/submission.py` and `scripts/predict.py` were changed.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_submission.py -q` passed all six generated-data tests. Ruff passed for `src/ie4483/submission.py`, `scripts/predict.py`, and `tests/test_submission.py`. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed all 41 tests on CPU.
## To change this
- To add another TTA view, change the inference block in `scripts/predict.py::predict` and keep probability rows aligned with their IDs; the cost is additional forward passes.
- To change the probability CSV columns, update `src/ie4483/submission.py::write_prediction_csvs` and its byte/content assertions in `tests/test_submission.py` together.
- To accept another class order, derive the dog index by name in `src/ie4483/submission.py::validate_predictions` and ensure the required label value remains dog=1.
## Rollback
Remove `src/ie4483/submission.py`, `scripts/predict.py`, `tests/test_submission.py`, the prediction instructions from `README.md`, and this record. The training and validation commands do not import the new module and remain independent.




<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:49 UTC for issue #9 attempt 9; compared HEAD..working tree (base c5e8428). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 0 component(s), +61/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +46/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/predict.py`, `src/ie4483/submission.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/submission.py`, `scripts/predict.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 1 tested in this run, 14 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/submission.py::validate_predictions` — output invariants (named code exists and changed)
  - [tested] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_submission.py -q`... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/submission.py::write_prediction_csvs` — CSV artifacts (named code exists and changed)
  - [checked] Components: `scripts/predict.py::predict` — run reconstruction and inference (named code exists and changed)
  - [checked] Components: `scripts/predict.py::main` — CLI (named code exists and changed)
  - [claimed] Components: `tests/test_submission.py` — generated test images and output checks (names no code)
  - [claimed] Key parameters: Batch size: `config.eval.batch_size` loaded by `scripts/predict.py::predict`; increasin... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: numeric sorting, duplicate IDs, empty test sets (valid header-only files), wro... (names no code)
  - [claimed] Key parameters: Worker count: saved training setting `extra["num_workers"]` loaded by `scripts/predict.... (no cited constant with a matching value)
  - [claimed] Key parameters: Flip TTA: `--tta flip` in `scripts/predict.py::main`; enabling it averages original and... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won
- WARN: references something that doesn't exist: config.yaml

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0009-test-set-prediction.md` gives the commit.
<!-- verified-facts:end -->
