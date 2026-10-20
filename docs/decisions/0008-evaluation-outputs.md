# 0008: Validation evaluation outputs
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #8 · Date: 2026-10-19
## In short
The project can now evaluate a completed training run against its labeled validation split and save machine-readable metrics and predictions alongside report-ready figures. The command also updates the project result contract at `results/latest.json`.
## How it works
1. The operator points the evaluation command at a run folder produced by training. That folder supplies both the resolved settings and the selected checkpoint, so the evaluator does not rely on a separate config that could drift.
2. The program reconstructs the same model, loads the selected weights, and reads only the labeled validation folders. It applies the deterministic evaluation image transform and computes class probabilities, predicted labels, and average cross-entropy.
3. It writes one row per validation image with its path, class IDs, and dog probability. The accompanying metrics file records accuracy, loss, sample count, and precision, recall, and F1 for each configured class.
4. The confusion matrix uses the configured class order. For each class, three contact sheets show the most confident correct cases, most confident incorrect cases, and images whose dog probability is closest to one half. Captions identify truth, prediction, and dog probability.
5. The latest-result file contains accuracy, the training seed, the validation split name, and the run folder name. Its location defaults to `results/latest.json` and can be supplied on the command line.
## Main design choice
The evaluator rebuilds the architecture from the run's saved config and reads the checkpoint's model state. Requiring users to supply a second config could make one-off testing convenient, but it allows preprocessing or architecture settings to diverge from the run being reported. The run folder already contains the authoritative configuration, so reconstructing from it makes evaluation repeatable. The code uses small direct metric calculations and matplotlib rather than adding a metric or plotting dependency; a library implementation would be preferable if the project later needs richer reports or confidence intervals.
## Traps: why the obvious fix is wrong
- The dog probability is column one only because the project class order is `[cat, dog]` (`configs/base.yaml::data.classes`). Changing that order requires updating the meaning of `p_dog` consistently.
- The validation split is labeled and used for reporting/model selection; `test/` is unlabeled and is not opened by this command.
- Uncertain examples are ranked by distance from probability 0.5, while correct/wrong examples are ranked by confidence in the predicted class. Ranking uncertainty by predicted-class confidence would select the opposite cases.
- The checkpoint is a training artifact dictionary. Loading only its top-level mapping as model parameters fails because state lives under its `model` field (`src/ie4483/train.py::_fit`).
## Components
- `src/ie4483/evaluate.py::evaluate` — run evaluation and artifact writes
- `src/ie4483/evaluate.py::_scores` — accuracy and per-class metrics
- `src/ie4483/evaluate.py::_gallery` — confidence-ranked report images
- `scripts/evaluate.py::main` — CLI entry point
- `tests/test_evaluate.py::test_evaluate_writes_validation_artifacts_and_result_contract` — generated run acceptance
## Breaks or needs follow-up
None. Evaluation adds an independent command and does not change training outputs.
## Problem and constraints
Rubric parts (d) and (e) require results that are easy to inspect and present. This is pipeline groundwork: validation uses generated images in tests, the checkpoint is created with `pretrained: false`, and no real-data score is claimed. Evaluation runs on CPU when CUDA is unavailable and does not fetch weights or data.
## Alternatives considered
- Pass a separate config path to evaluation — rejected because it can disagree with the config saved alongside `best.pt`.
- Depend on scikit-learn metric and plotting helpers — rejected because simple binary metrics and a small matrix plot do not require another runtime dependency.
## Trade-offs
Each run emits six class-by-selection gallery files (`src/ie4483/evaluate.py::evaluate`), even when no matching images exist; placeholder figures keep the artifact set predictable. Evaluation holds predictions in memory before writing artifacts, which is suitable for the current validation set but could be streamed for much larger datasets. The `results/latest.json` default is relative to the working directory, which makes command invocation location observable; `--results` allows a chosen location.
## Key parameters
- Gallery limit 5 (`src/ie4483/evaluate.py::_GALLERY_LIMIT`): keeps each report sheet compact; raising it adds images and width, lowering it shows fewer examples.
- Result path `results/latest.json` (`scripts/evaluate.py::main`): provides the requested default handoff; `--results` changes the destination.
- Class order `[cat, dog]` (`configs/base.yaml::data.classes`): determines metric names, confusion-matrix axes, and the dog-probability column; changing order requires preserving the dog semantic explicitly.
## Assumptions
The run folder contains `config.yaml` and a compatible `best.pt` produced by the project trainer. The configured class list contains cat at index zero and dog at index one, as specified by project data conventions. Input image paths remain readable while the report is generated.
## Edge cases
- Handled: an empty readable validation set raises a clear `ValueError`; a missing gallery category still gets a placeholder image; absent predicted or true classes receive zero-valued precision/recall/F1 where undefined.
- Not handled: corrupt or incompatible checkpoints and missing image files surface the underlying loader error. The caller should use the matching run folder and complete validation data.
## Not verified
No real dataset, cached pretrained weights, or CUDA device was used. The generated-data test verifies the artifact contract and CSV-derived accuracy, but not the visual quality of the galleries.
## Changes outside the scope
- `README.md`: documents the new evaluation command as required for newly added commands.
- `tests/test_evaluate.py`: covers the generated-run behavior required by project instructions.
- `docs/decisions/0008-evaluation-outputs.md`: required decision record.
These supporting files are outside the literal production scope paths. No other production module was changed.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_evaluate.py -q` passed the generated tiny-run artifact test. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 35 tests on CPU. Ruff passed for the evaluator, CLI, and acceptance test.
## To change this
- To show more or fewer examples per gallery, change `src/ie4483/evaluate.py::_GALLERY_LIMIT`; more examples enlarge the image, fewer reduce coverage.
- To move the default result handoff, change the `--results` default in `scripts/evaluate.py::main`; callers relying on `results/latest.json` must update their path.
- To support a different class order, derive dog probability by class name rather than index and verify labels and gallery names against the dataset mapping.
## Rollback
Remove `src/ie4483/evaluate.py` and `scripts/evaluate.py`, remove the evaluation instructions from `README.md`, remove `tests/test_evaluate.py`, and remove this record. Training checkpoints remain usable by the existing training workflow; no earlier module imports the evaluator.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:42 UTC for issue #8 attempt 8; compared HEAD..working tree (base 3397192). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 0 component(s), +19/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +130/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/evaluate.py`, `src/ie4483/evaluate.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/evaluate.py`, `scripts/evaluate.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

5 checked by the tool, 0 tested in this run, 12 claimed only (of 17 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 1 claimed
- Key parameters: 1 checked, 0 tested, 2 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/evaluate.py::evaluate` — run evaluation and artifact writes (named code exists and changed)
  - [checked] Key parameters: Gallery limit 5 (`src/ie4483/evaluate.py::_GALLERY_LIMIT`): keeps each report sheet com... (value matches the code: _GALLERY_LIMIT = 5)
  - [checked] Components: `src/ie4483/evaluate.py::_scores` — accuracy and per-class metrics (named code exists and changed)
  - [checked] Components: `src/ie4483/evaluate.py::_gallery` — confidence-ranked report images (named code exists and changed)
  - [checked] Components: `scripts/evaluate.py::main` — CLI entry point (named code exists and changed)
  - [claimed] Components: `tests/test_evaluate.py::test_evaluate_writes_validation_artifacts_and_result_contract`... (names no code)
  - [claimed] Key parameters: Result path `results/latest.json` (`scripts/evaluate.py::main`): provides the requested... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: an empty readable validation set raises a clear `ValueError`; a missing galler... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_evaluate.py -q` pa... (no test named)
  - [claimed] Key parameters: Class order `[cat, dog]` (`configs/base.yaml::data.classes`): determines metric names,... (no cited constant with a matching value)
  - full per-claim list (17) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: references something that doesn't exist: config.yaml

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0008-evaluation-outputs.md` gives the commit.
<!-- verified-facts:end -->
