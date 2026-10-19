# 0018: Training-schedule and loss tidy-ups from the pipeline review
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #18 · Date: 2026-10-19
## In short
Warmup cosine scheduling now spans only the epochs after warmup, so its final step reaches the minimum at the end of the run. Smoothed focal loss now focuses using the model's probability for the true class, and the model card states that `best.pt` follows validation accuracy while early stopping follows validation loss.
## How it works
1. Training raises the learning rate linearly during configured warmup epochs, then advances cosine annealing for the remaining epochs. The schedule's final update occurs after the final epoch's validation pass.
2. The selected loss computes cross-entropy per example, including configured label smoothing and optional class weights. For positive focal gamma, it multiplies that value by a factor derived from the softmax probability assigned to the actual target class.
3. A gamma of zero returns the mean cross-entropy directly, preserving the neutral focal setting. The run continues to replace the best checkpoint on validation-accuracy improvement and to count validation-loss non-improvements for early stopping.
4. Model-card generation writes both selection rules beside the configured schedule and model details, so readers can distinguish the saved model criterion from the stopping criterion.
## Main design choice
For warmup, keep scheduler stepping after each eligible epoch and set its horizon to `epochs - warmup_epochs` (`src/ie4483/train.py::_fit`). Moving the scheduler step before validation/training or changing warmup ordering would alter which learning rate each epoch uses and complicate resume state; correcting only the horizon preserves the existing trajectory up to the endpoint. For focal loss, use `softmax(logits)[target]` while retaining smoothed cross-entropy as the loss term (`src/ie4483/imbalance.py::FocalLoss`). Deriving the focusing probability from `exp(-ce)` is shorter, but with label smoothing it is no longer the model probability of the target class. A second unsmoothed CE could also recover that probability only absent weights and other reductions, so direct softmax gather is clearer and independent of those loss settings.
## Traps: why the obvious fix is wrong
- `exp(-smoothed_ce)` is not the true-class probability. Label smoothing combines losses for target and non-target classes, so that shortcut makes focal weighting depend on smoothing.
- The scheduler is stepped after validation at the end of an epoch, and warmup epochs do not step it. Its horizon must exclude warmup epochs or the minimum is delayed beyond the configured run.
- The model-selection and stopping criteria intentionally differ; matching them would change established training behavior.
## Components
- `src/ie4483/train.py::_fit` — cosine horizon
- `src/ie4483/imbalance.py::FocalLoss` — target probability
- `src/ie4483/figures.py::write_model_card` — selection text
- `tests/test_train.py` — final schedule state
- `tests/test_imbalance.py` — smoothed and zero-gamma loss
- `tests/test_figures.py` — card selection rule
## Breaks or needs follow-up
None. Defaults and checkpoint formats are unchanged.
## Problem and constraints
The pipeline review found an endpoint mismatch in warmup cosine scheduling, an incorrect focal factor under label smoothing, and an undocumented difference between checkpoint selection and early stopping. Changes stay in the declared training, imbalance, figure, and model-figure scope; tests and this record provide required supporting coverage. No real data, downloaded weights, or GPU are needed.
## Alternatives considered
- Keep `T_max=epochs` and merely document the late minimum — leaves the schedule bug in place.
- Compute focal confidence through smoothed CE — conflates target probability with the smoothed objective.
- Add only general prose to the card — rejected in favor of one explicit selection-rule line with both criteria.
## Trade-offs
The schedule correction changes learning rates in runs with warmup by reaching the minimum on the configured final epoch; prior runs with warmup used a longer cosine tail. Focal loss values with positive gamma and smoothing will change to reflect target confidence directly. No measurable model-quality claim is made because this is pipeline groundwork and no real-data experiment was run.
## Key parameters
- `T_max=max(1, epochs - warmup_epochs)` (`src/ie4483/train.py::_fit`): increasing the post-warmup duration spreads the cosine decay; decreasing it reaches the minimum sooner. The lower bound keeps the scheduler valid when warmup consumes all configured epochs.
- `gamma` (`src/ie4483/imbalance.py::FocalLoss`): zero selects plain mean CE; larger positive values strengthen down-weighting of examples with high true-class probability.
- `label_smoothing` (`src/ie4483/imbalance.py::FocalLoss`): controls the CE target distribution, not the probability used by the focusing term.
## Assumptions
The existing scheduler step order and epoch numbering are intentional, and `warmup_epochs` is a nonnegative configured count. The training config already validates its values. The model-card selection line describes training behavior even if a run stops early.
## Edge cases
- Handled: four epochs with one warmup epoch finish with `T_max=3` and scheduler LR at its minimum; focal loss with smoothing matches a hand-computed target-probability factor; gamma zero equals CE.
- Not handled: if warmup is at least the entire epoch count, the cosine horizon is clamped to one, but no post-warmup epoch may run the schedule. This issue does not redefine warmup validation or semantics.
## Not verified
CUDA training, real-data learning curves, and the effect of this schedule/loss correction on final accuracy were not measured.
## Changes outside the scope
`tests/test_train.py`, `tests/test_imbalance.py`, `tests/test_figures.py`, and this decision record are supporting files required by the project instructions. No production file outside the declared Scope changed; README needs no update because no command or config key was added.
## How it was verified
Focused CPU run `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_train.py tests/test_imbalance.py tests/test_figures.py`: 14 passed. Full CPU run `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q`: 69 passed in 145.67 seconds. Ruff check on the three scoped Python modules and three touched test files passed. The scheduler regression uses a mocked epoch and emits PyTorch's scheduler-before-optimizer warning because that mock intentionally performs no optimizer step; production `_epoch` performs optimization.
## To change this
To use a different warmup policy, change warmup LR assignment and scheduler stepping together in `src/ie4483/train.py::_fit`; changing only one can skip or repeat schedule steps. To alter focal strength, tune `train.focal_gamma`; to alter smoothing, tune `train.label_smoothing`. Keep the focal factor based on the unsmoothed softmax target probability unless the intended objective explicitly changes.
## Rollback
Revert the `T_max` expression, focal probability calculation, model-card line, matching regression tests, and this record together. Check that the training and imbalance tests reflect the older intended behavior before restoring it; no artifacts or migrations depend on the changes.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:42 UTC for issue #18 attempt 5; compared HEAD..working tree (base 39181ea). Records are claims to check against the code, not proof.

**Change summary**

- `src/ie4483/`: 3 file(s), 3 component(s), +6/-2 lines
- plus 3 test file(s)

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/train.py`, `src/ie4483/imbalance.py`, `src/ie4483/figures.py`, `scripts/model_figure.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

3 checked by the tool, 0 tested in this run, 14 claimed only (of 17 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 3 checked, 0 tested, 3 claimed
- Key parameters: 0 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 3 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/train.py::_fit` — cosine horizon (named code exists and changed)
  - [checked] Components: `src/ie4483/imbalance.py::FocalLoss` — target probability (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::write_model_card` — selection text (named code exists and changed)
  - [claimed] Components: `tests/test_train.py` — final schedule state (names no code)
  - [claimed] Key parameters: `T_max=max(1, epochs - warmup_epochs)` (`src/ie4483/train.py::_fit`): increasing the po... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: four epochs with one warmup epoch finish with `T_max=3` and scheduler LR at it... (names no code)
  - [claimed] How it was verified: Focused CPU run `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test... (no test named)
  - [claimed] Components: `tests/test_imbalance.py` — smoothed and zero-gamma loss (names no code)
  - [claimed] Key parameters: `gamma` (`src/ie4483/imbalance.py::FocalLoss`): zero selects plain mean CE; larger posi... (no cited constant with a matching value)
  - [claimed] Components: `tests/test_figures.py` — card selection rule (names no code)
  - full per-claim list (17) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0018-training-schedule-loss-tidy-ups.md` gives the commit.
<!-- verified-facts:end -->
