# 0013: CIFAR-10 class imbalance and remedies
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #13 · Date: 2026-10-19
## In short
CIFAR-10 runs can now retain seeded per-class fractions of the training split and record the resulting class counts. Training supports inverse-frequency class weights, balanced weighted sampling and focal loss, while evaluation reports macro recall and macro-F1 alongside accuracy.
## How it works
1. Training reads the established CIFAR training portion and validation portion. Only the training portion is eligible for subsampling; validation and the official test batch are loaded through their existing paths.
2. A configured fraction is applied independently to each selected class. Unlisted classes keep every training example. The seeded selection is without replacement, and its class counts are written to the run directory using class names.
3. The run can calculate normalized inverse-frequency weights from the selected labels, draw examples with replacement using inverse class frequency, or apply focal loss. These switches can be combined; otherwise the existing cross-entropy path remains active.
4. Validation continues to select the saved checkpoint. Evaluation then calculates each class's recall and F1 and averages each metric across classes, so minority classes contribute equally to macro metrics.
5. The imbalance sweep varies the retained fraction and remedy switches over its configured seeds. Runs continue to use the standard run-folder artifact layout.
## Main design choice
`src/ie4483/imbalance.py::imbalanced_subset` returns a `Subset` over the existing CIFAR dataset, rather than modifying `src/ie4483/cifar.py::CIFAR10Dataset` to have a second concept of train data. This leaves its deterministic balanced validation split and test path intact, and keeps subset selection local to training. A specialized imbalanced CIFAR dataset was possible but would entangle label distribution changes with the established data adapter. Remedies are explicit training switches sharing one loop; separate training loops would duplicate checkpointing, scheduling and AMP behavior.
## Traps: why the obvious fix is wrong
- Do not subsample validation or test data: metrics would stop describing the balanced evaluation distribution required by the task.
- Class weights and sampler frequencies must be computed after subsampling; computing them before would compensate for the wrong distribution.
- Weighted sampling uses replacement. Without replacement, a full epoch over all selected items cannot balance unequal class counts.
- Macro metrics must average per-class values, rather than pool all examples; pooled metrics let common classes dominate.
- Focal gamma zero must reduce to cross-entropy; otherwise the switch changes the loss unexpectedly at its neutral setting.
## Components
- `src/ie4483/imbalance.py::imbalanced_subset` — seeded selection
- `src/ie4483/imbalance.py::class_weights` — normalized inverse frequency
- `src/ie4483/imbalance.py::balanced_sampler` — replacement sampling
- `src/ie4483/imbalance.py::FocalLoss` — focal criterion
- `src/ie4483/train.py::train` — data and remedy switches
- `src/ie4483/evaluate.py::_scores` — macro metrics
- `configs/cifar10.yaml` — disabled-by-default switches
- `configs/sweeps/imbalance.yaml` — remedy sweep
- `tests/test_imbalance.py` — acceptance coverage
## Breaks or needs follow-up
None. Defaults preserve the existing training distribution and loss. The extra metric keys extend evaluation JSON without removing existing fields.
## Problem and constraints
The report study needs a reproducible way to reduce selected CIFAR class counts and compare at least two remedies. Changes must remain local-data-only, deterministic, CPU-capable and independent of pretrained weights for tests.
## Alternatives considered
- Alter the CIFAR adapter itself — rejected because the imbalanced training view should not change its balanced validation/test semantics.
- Train a separate focal/sampler loop — rejected because it duplicates the existing optimizer and checkpoint logic.
- Use pooled F1/recall — rejected because majority classes would dominate the requested imbalance analysis.
## Trade-offs
Sampling a subset adds an in-memory index list, while weighted sampling can repeat minority examples within an epoch. Combining class weights and weighted sampling may overcompensate; the sweep makes that interaction observable, but does not choose a best remedy without real-data runs.
## Key parameters
- `train.imbalance_fractions` (`configs/cifar10.yaml`): mapping from label ID to retained fraction; lower values remove more examples, and absent labels retain all examples.
- `train.class_weighted` (`configs/cifar10.yaml`): computes weights as total-count divided by class-count times number of classes, then normalizes mean weight to 1 (`src/ie4483/imbalance.py::class_weights`).
- `train.weighted_sampler` (`configs/cifar10.yaml`): samples `len(training subset)` examples with replacement (`src/ie4483/imbalance.py::balanced_sampler`); larger epochs could reduce sampling noise but cost more compute.
- `train.focal_gamma` (`configs/cifar10.yaml`): `null` disables focal loss; zero is cross-entropy, and higher values down-weight easy examples more (`src/ie4483/imbalance.py::FocalLoss`).
- Fractions must be greater than 0 and at most 1 (`src/ie4483/imbalance.py::imbalanced_subset`); a positive fraction retains at least one item per represented class.
## Assumptions
`CIFAR10Dataset.labels` contains official integer IDs and the configured class list uses that ID order. If class order is changed, the named count artifact and model output interpretation can diverge.
## Edge cases
- Handled: zero fractions, fractions over 1, and negative fractions raise; absent fraction entries preserve full class counts; empty classes are rejected when computing weights or a sampler.
- Not handled: a configuration can enable class weighting when `class_weights` are also manually supplied; generated weights take precedence.
## Not verified
No real CIFAR data, pretrained weights or GPU run was used. The sweep was not executed end-to-end because it is an experiment configuration and would require real training runs.
## Changes outside the scope
- `src/ie4483/evaluate.py::_scores`: required to expose macro recall/F1 in evaluator output, because the issue's scope names training but not the evaluator module.
- `README.md`: required because this change adds config keys and a sweep command.
- `tests/test_imbalance.py`: required acceptance tests for the new behaviors.
## How it was verified
Focused CPU tests: `tests/test_imbalance.py`, `tests/test_evaluate.py` and `tests/test_train.py` passed (8 tests). Full CPU suite `python -m pytest -q` passed (50 tests). Ruff passed on the changed Python files.
## To change this
To retain more or fewer minority samples, change the relevant label fraction in `configs/cifar10.yaml::train.imbalance_fractions`; larger fractions reduce information loss. To compare a different combination, edit `configs/sweeps/imbalance.yaml::parameters`; more combinations multiply run count across its seed list. To change focal behavior, adjust `train.focal_gamma`; larger values focus more strongly on hard examples.
## Rollback
Remove `src/ie4483/imbalance.py`, its training integration, CIFAR imbalance keys and sweep, then remove the macro fields in `src/ie4483/evaluate.py::_scores` and the README section. Existing non-imbalance training defaults remain compatible.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 03:23 UTC for issue #13 attempt 13; compared HEAD..working tree (base ef81a6a). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +4/-0 lines
- `configs/sweeps/`: 1 file(s), 0 component(s), +8/-0 lines
- `src/ie4483/`: 3 file(s), 4 component(s), +97/-3 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `configs/sweeps/imbalance.yaml`, `src/ie4483/imbalance.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/imbalance.py`, `src/ie4483/train.py`, `configs/cifar10*.yaml`, `configs/sweeps/**`. Verdict: violation.
- out of scope: `src/ie4483/evaluate.py::_scores` (modified); explained in the record: yes

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

8 checked by the tool, 1 tested in this run, 16 claimed only (of 25 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 8 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 5 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/imbalance.py::imbalanced_subset` — seeded selection (named code exists and changed)
  - [tested] How it was verified: Focused CPU tests: `tests/test_imbalance.py`, `tests/test_evaluate.py` and `tests/test_... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/imbalance.py::class_weights` — normalized inverse frequency (named code exists and changed)
  - [checked] Components: `src/ie4483/imbalance.py::balanced_sampler` — replacement sampling (named code exists and changed)
  - [checked] Components: `src/ie4483/imbalance.py::FocalLoss` — focal criterion (named code exists and changed)
  - [checked] Components: `src/ie4483/train.py::train` — data and remedy switches (named code exists and changed)
  - [checked] Components: `src/ie4483/evaluate.py::_scores` — macro metrics (named code exists and changed)
  - [checked] Components: `configs/cifar10.yaml` — disabled-by-default switches (named code exists and changed)
  - [checked] Components: `configs/sweeps/imbalance.yaml` — remedy sweep (named code exists and changed)
  - [claimed] Components: `tests/test_imbalance.py` — acceptance coverage (names no code)
  - full per-claim list (25) in the JSON record

**Consistency**

0 error(s), 0 warning(s) between the rationale above and the change.

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0013-cifar10-class-imbalance.md` gives the commit.
<!-- verified-facts:end -->
