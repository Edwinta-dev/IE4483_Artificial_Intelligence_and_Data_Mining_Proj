# 0016: Resume interrupted training from the last checkpoint
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #16 · Date: 2026-10-19
## In short
Training now writes an atomic `last.pt` checkpoint after every completed epoch and can continue that run through the CLI or experiment runners. Resume verifies the resolved configuration and restores model and optimization state, early-stopping state, and random state so the next epoch follows the same path as uninterrupted training.
## How it works
1. Start training normally. The run folder records its resolved configuration and epoch metrics, and each completed epoch updates `last.pt`; `best.pt` continues to hold the checkpoint selected by validation accuracy.
2. If training stops between epochs, run the training command with `--resume` pointing at that run folder. The supplied config and overrides are resolved as usual and compared with the saved config before training starts.
3. The trainer rebuilds the model, optimizer, schedule, scaler, and data loaders, then restores their saved states and the Python, NumPy, PyTorch, CUDA, and shuffle-generator random states. It opens the existing metrics file in append mode and continues at the following epoch.
4. Seed and sweep runners look for a run folder with `last.pt` when validation evaluation has not produced its completion marker. They pass that folder back to training, then evaluate and summarize it when training finishes.
## Main design choice
The checkpoint contains the complete live training state, including the data-loader generator state, while `best.pt` remains a separate model-selection artifact. Reinitializing from `best.pt` and restarting would be simpler and smaller, but it would lose optimizer momentum, schedule position, stale-loss patience, and the next shuffled sample order. Keeping both state bundles in one atomic `last.pt` makes each epoch boundary a coherent restart point. The trade-off is a larger checkpoint, especially when the backbone is trainable; saving only model weights would be better if storage were the limiting constraint and exact continuation were unnecessary.
## Traps: why the obvious fix is wrong
Restoring only global RNG state is insufficient: the shuffled loader has its own generator, so it must also be restored. `best.pt` is not a continuation checkpoint because it omits optimizer, scheduler, scaler, and patience state. Metrics append only on explicit resume, preventing duplicate headers or epoch rows. A mismatch is rejected rather than silently training under a different recipe in a run folder.
## Components
- `src/ie4483/train.py::_fit` — state restore/save
- `src/ie4483/train.py::train` — run reuse and config check
- `scripts/train.py::main` — `--resume`
- `src/ie4483/experiments.py::_partial_run` — unfinished-run lookup
- `tests/test_train.py` — interrupted equivalence
- `tests/test_experiments.py` — sweep resume
## Breaks or needs follow-up
None. Existing training calls still create fresh run directories; resume is opt-in.
## Problem and constraints
Long training can be stopped by laptop throttling or an operator. Recovery must retain the training trajectory without touching real project data, requiring a GPU, or downloading weights. Tests use generated image folders and CPU execution. Production changes stay within the declared training, CLI, and experiment-runner scope; README and tests are required supporting files.
## Alternatives considered
- Resume from `best.pt`: omits optimizer and stopping state, so later steps diverge.
- Replay earlier epochs from the initial seed: requires repeated work and cannot recover an interruption if the initial environment changed.
## Trade-offs
Per-epoch atomic writes add checkpoint I/O and `last.pt` duplicates a model state already represented by `best.pt`. Recovery is limited to completed epoch boundaries; work in an interrupted epoch is repeated. A run must remain intact, including its config, metrics, data, and feature cache if applicable.
## Key parameters
`last.pt` is written after every completed epoch (`src/ie4483/train.py::_fit`); writing less often reduces I/O but loses more work. `--resume RUN_DIR` (`scripts/train.py::main`) selects the existing run; any resolved config difference raises `ValueError` to protect run provenance.
## Assumptions
The saved run folder and data remain available and unchanged. Resume assumes the same software and hardware-compatible optimizer state; config equality cannot detect changed image contents at the same paths.
## Edge cases
- Handled: missing checkpoint and config mismatch produce errors; completed epochs are not duplicated in metrics.
- Not handled: corrupt checkpoint, changed data at stable paths, or interruption during an epoch; these surface as load errors, altered inputs, or a replay of that epoch.
## Not verified
CUDA RNG restoration and AMP recovery were not exercised. The test interruption occurs at the epoch checkpoint boundary, not during serialization or a GPU kernel.
## Changes outside the scope
- `README.md`: documents the new resume command and runner behavior.
- `tests/test_train.py` and `tests/test_experiments.py`: required regression coverage.
- `docs/decisions/0016-resume-interrupted-training.md`: required decision record.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_train.py tests/test_experiments.py` passed 8 tests in 35.99 seconds. The interrupted CPU run matched straight-through metrics and best weights within tolerance; the test also rejected a changed learning rate. The full CPU suite passed 65 tests in 153.44 seconds. Ruff initially reported import-order issues, which were corrected before final verification.
## To change this
To reduce checkpoint I/O, move the `_atomic_save` call in `src/ie4483/train.py::_fit` behind an epoch interval; recovery then loses all later epochs since the previous save. To retain more mid-epoch work, add sampler position and in-epoch batch state, with substantially more recovery complexity.
## Rollback
Remove resume handling from `src/ie4483/train.py`, `scripts/train.py`, and `src/ie4483/experiments.py`, then remove the new tests, README usage note, and this record. Existing `best.pt` artifacts remain readable; `last.pt` files are only used by this feature.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:25 UTC for issue #16 attempt 3; compared HEAD..working tree (base 92e791f). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 1 component(s), +2/-1 lines
- `src/ie4483/`: 2 file(s), 9 component(s), +77/-19 lines
- plus 2 test file(s), 1 doc file(s)

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/train.py`, `scripts/train.py`, `src/ie4483/experiments.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 0 tested in this run, 9 claimed only (of 13 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 1 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 1 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/train.py::_fit` — state restore/save (named code exists and changed)
  - [checked] Components: `src/ie4483/train.py::train` — run reuse and config check (named code exists and changed)
  - [checked] Components: `scripts/train.py::main` — `--resume` (named code exists and changed)
  - [checked] Components: `src/ie4483/experiments.py::_partial_run` — unfinished-run lookup (named code exists and changed)
  - [claimed] Components: `tests/test_train.py` — interrupted equivalence (names no code)
  - [claimed] Key parameters: `last.pt` is written after every completed epoch (`src/ie4483/train.py::_fit`); writing... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing checkpoint and config mismatch produce errors; completed epochs are no... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_train.py tests/... (no test named)
  - [claimed] Components: `tests/test_experiments.py` — sweep resume (names no code)
  - full per-claim list (13) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 4 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0016-resume-interrupted-training.md` gives the commit.
<!-- verified-facts:end -->
