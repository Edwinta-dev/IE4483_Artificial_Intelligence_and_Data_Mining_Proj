# 0015: VRAM probe for safe training batches
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #15 · Date: 2026-10-19
## In short
The repository now has a helper that searches for the largest training batch that completes and returns a safety-adjusted batch size. A CUDA command benchmarks all supported ResNet and freeze mode combinations with synthetic data, writes a JSON report, and can update supplied configs.
## How it works
1. Run the probe with an optional config path. Its image size comes from that config; without one, it uses 224 pixels (`scripts/probe_vram.py::main`). Configs can be repeated when they share an image size.
2. The command seeds random initialization and generated inputs (42 by default, `scripts/probe_vram.py::main`) through `src/ie4483/seed.py::seed_everything`. For each of ResNet-18 and ResNet-50 (`scripts/probe_vram.py::ARCHITECTURES`), it builds frozen, last-block, and full training variants (`scripts/probe_vram.py::MODES`) with pretrained weights disabled. It creates random images and labels, performs an autocast forward pass, cross-entropy loss, backward pass, and AdamW step.
3. The search tries increasing batch sizes up to its configured limit. A CUDA out-of-memory error marks an attempt as too large; the CUDA cache is emptied after each try. The largest successful batch is multiplied by the safety fraction (0.8 by default, `src/ie4483/vram.py::find_max_batch`) and rounded down, with a minimum of one when any batch succeeds.
4. The JSON records that safe batch, the largest successful batch, peak allocated bytes from that successful step, image size, and gradient accumulation needed to reach at least the target effective batch of 32 (`src/ie4483/vram.py::EFFECTIVE_BATCH_SIZE`).
5. With `--write-configs`, each supplied YAML file gets the measured batch and accumulation values for its own architecture and freeze mode. Without CUDA the command prints a message and exits without probing.
## Main design choice
The probe performs a synthetic optimizer step through `src/ie4483/model.py::build_model`, rather than measuring only model construction or a forward pass. Forward-only memory omits training activations retained for gradients and optimizer state, so it would not answer this issue's full fine-tuning risk. A full data-loader run would include augmentation and host memory behavior, but would require project data and add unrelated variability. The synthetic step is deterministic in shape and needs no data or weights; input values are not intended as model training data.
## Traps: why the obvious fix is wrong
- A forward-only capacity check misses backward activations and optimizer allocations, so it can recommend a batch that fails during training.
- A failed step can leave cached CUDA blocks; `src/ie4483/vram.py::find_max_batch` empties the cache after successes and OOM failures.
- Probe weights are deliberately random (`pretrained: false`). Loading weights would add a network/cache dependency and is not needed to measure activation shapes.
- A mode's capacity depends on which backbone parameters need gradients. The probe passes explicit `freeze_mode` values to the existing model factory, even when the YAML schema only supports the legacy frozen/full boolean.
- Safety is applied to batch count, not as a percentage of VRAM bytes. A nonlinear jump in activation or optimizer memory can make a batch fraction conservative by less or more than the same fraction of memory.
## Components
- `src/ie4483/vram.py::find_max_batch` — bounded search and cache cleanup
- `src/ie4483/vram.py::accumulation_steps` — target batch arithmetic
- `src/ie4483/vram.py::write_probe_json` — report writer
- `scripts/probe_vram.py::probe` — six synthetic benchmarks
- `scripts/probe_vram.py::_write_configs` — opt-in config update
- `tests/test_vram.py` — mocked search, math, JSON, config and CPU path
## Breaks or needs follow-up
None known. Probe measurements depend on the GPU, driver, PyTorch build, and image size; rerun on the training machine after changing any of them.
## Problem and constraints
Full ResNet-50 fine-tuning stores intermediate activations and gradients that frozen feature extraction does not. The team's target laptop has 4 GB of VRAM, while pipeline tests must use generated inputs and run without requiring CUDA. The helper is hardware-independent under a fake step function; the command is CUDA-only and makes no real-data or weight requests.
## Alternatives considered
- Estimate capacity from a fixed memory-per-image value: rejected because freeze modes and backbone architectures have different activation and optimizer costs.
- Benchmark only a forward pass: rejected because the measured training path includes backward and optimizer steps.
- Add `freeze_mode` fields to `src/ie4483/config.py`: outside the issue's source Scope; explicit modes remain a probe concern and existing config behavior is unchanged.
## Trade-offs
The bounded search can under-report the true maximum when its limit (default 128, `scripts/probe_vram.py::main`) is too low. Synthetic inputs isolate model-step allocations but do not account for data-loader workers, augmentation buffers, competing GPU processes, or fragmentation that may occur during a long job. The safety fraction (`src/ie4483/vram.py::find_max_batch`) reduces batch count uniformly and should be treated as a practical headroom setting, not a guarantee against all runtime OOMs.
## Key parameters
- Safety fraction `0.8` (`src/ie4483/vram.py::find_max_batch`): scales the largest successful batch down; lower values add more batch-count headroom and increase accumulation needs.
- Initial batch `1` and limit `128` (`scripts/probe_vram.py::main`): CLI defaults bound the search; raising the limit may find a larger fit but can cause expensive failed attempts.
- Effective batch `32` (`src/ie4483/vram.py::EFFECTIVE_BATCH_SIZE`): accumulation uses ceiling division, so effective microbatch count is at least 32 images when the safe batch is below 32.
- Seed `42` (`scripts/probe_vram.py::main`): fixes model initialization and synthetic inputs; changing it changes sampled values, not tensor shapes.
- AMP enabled for CUDA (`scripts/probe_vram.py::_training_step`): matches the configured training path; disabling it would measure a different memory profile.
## Assumptions
The main capacity variable under study is the model training step at the configured image size. The CUDA runtime reports recoverable failures as `torch.cuda.OutOfMemoryError`. Config updates should use each config's selected architecture and its explicit freeze mode, or existing `freeze_backbone` compatibility value.
## Edge cases
- Handled: a first-attempt OOM returns safe batch 0 and no accumulation count; config updates skip such results. Start/limit and safety values are validated.
- Handled: CUDA unavailability exits before attempting a benchmark; the CPU test mocks this condition.
- Not handled: configs with different image sizes cannot share one result set; the CLI reports a parser error. A concurrent GPU workload can make the result stale immediately after probing.
## Not verified
The CUDA command ran on the available machine and completed the six synthetic 224-pixel cases, but the machine was not verified as the target 4 GB RTX 3050. The benchmark did not measure data-loader memory or stability across a full training run. Config mutation was tested via its helper, not against the actual project YAML files.
## Changes outside the scope
- `README.md`: documents the new command as required by project instructions.
- `tests/test_vram.py`: adds the required regression coverage.
- `docs/decisions/0015-vram-probe.md`: required decision record.
## How it was verified
- `tests/test_vram.py` exercised threshold OOM search, custom margin, max-at-limit behavior, accumulation arithmetic, JSON round trip, config update, and no-CUDA CLI behavior.
- Full `pytest -q` completed with 64 passing tests.
- `ruff check` passed for the implementation and new test files.
- `scripts/probe_vram.py` ran on the available CUDA machine and wrote its report under `runs/`; that generated output was removed and is not part of the repository change.
## To change this
- To increase safety, lower `safety_margin` from 0.8 in `src/ie4483/vram.py::find_max_batch`; the cost is more optimizer steps per effective batch.
- To search higher capacities, raise the CLI `--limit` default in `scripts/probe_vram.py::main`; the cost is potentially longer probes and larger failed allocations.
- To change effective batch, change `src/ie4483/vram.py::EFFECTIVE_BATCH_SIZE`; config accumulation values will change accordingly.
## Rollback
Remove `src/ie4483/vram.py`, `scripts/probe_vram.py`, and `tests/test_vram.py`, revert the README command section and config comment, and remove this record. Training does not import the new helper, so reverting the probe does not affect existing training runs.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 07:14 UTC for issue #15 attempt 2; compared HEAD..working tree (base c542ad4). Records are claims to check against the code, not proof.

**Change summary**

- `configs/`: 1 file(s), 0 component(s), +1/-1 lines
- `scripts/`: 1 file(s), 0 component(s), +136/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +88/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/probe_vram.py`, `src/ie4483/vram.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/probe_vram.py`, `src/ie4483/vram.py`, `configs/**`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

6 checked by the tool, 1 tested in this run, 19 claimed only (of 26 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 5 checked, 0 tested, 1 claimed
- Key parameters: 1 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 2 claimed
- How it was verified: 0 checked, 1 tested, 3 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/vram.py::find_max_batch` — bounded search and cache cleanup (named code exists and changed)
  - [checked] Key parameters: Effective batch `32` (`src/ie4483/vram.py::EFFECTIVE_BATCH_SIZE`): accumulation uses ce... (value matches the code: EFFECTIVE_BATCH_SIZE = 32)
  - [tested] How it was verified: `tests/test_vram.py` exercised threshold OOM search, custom margin, max-at-limit behavi... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/vram.py::accumulation_steps` — target batch arithmetic (named code exists and changed)
  - [checked] Components: `src/ie4483/vram.py::write_probe_json` — report writer (named code exists and changed)
  - [checked] Components: `scripts/probe_vram.py::probe` — six synthetic benchmarks (named code exists and changed)
  - [checked] Components: `scripts/probe_vram.py::_write_configs` — opt-in config update (named code exists and changed)
  - [claimed] Components: `tests/test_vram.py` — mocked search, math, JSON, config and CPU path (names no code)
  - [claimed] Key parameters: Safety fraction `0.8` (`src/ie4483/vram.py::find_max_batch`): scales the largest succes... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: a first-attempt OOM returns safe batch 0 and no accumulation count; config upd... (names no code)
  - full per-claim list (26) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 8 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: configs/base.yaml (modified) changed but not mentioned in the record

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0015-vram-probe.md` gives the commit.
<!-- verified-facts:end -->
