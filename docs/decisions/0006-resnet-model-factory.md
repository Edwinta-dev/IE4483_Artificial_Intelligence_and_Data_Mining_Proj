# 0006: ResNet model factory with custom head and freeze modes
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #6 · Date: 2026-10-19
## In short
The package now builds ResNet-18 or ResNet-50 with a replaceable classifier head, optional dropout and hidden layer, and three backbone training modes. It reports input/output shapes, backbone feature width, and parameter totals; pretrained initialization fails clearly if its checkpoint is not already cached.
## How it works
1. The caller supplies the project config, a model config, or a mapping. The factory reads the architecture, whether to use cached ImageNet weights, class count, head options, and freeze mode. The existing config's `freeze_backbone` flag remains supported as a compatibility default.
2. It builds the selected torchvision ResNet. Pretrained mode checks the expected file in torch's hub checkpoint directory before asking torchvision to load it, preventing an implicit network fetch.
3. The original classifier is removed. The pooled feature vector passes through the optional dropout, optional learned hidden layer and ReLU, then a final linear layer producing one score per class.
4. In frozen mode only the new head's parameters can update; in last-block mode layer4 and the head can update; in full mode all parameters can update. BatchNorm layers in frozen sections return to evaluation mode when training is enabled.
5. The summary runs one zero-valued image through the model, restores its prior train/eval state, and reports the shapes and parameter counts. Tests build both backbones without weights on CPU.
## Main design choice
The factory accepts the existing config shape and optional values without extending the strict typed config schema, because `config.py` is outside this issue's declared scope. Extending that schema would make options easier to validate and expose in YAML, but requires an additional in-scope file change that the issue does not authorize. A wrapper module holds the ResNet feature extractor and head explicitly, rather than mutating torchvision's `fc` into a multi-stage `Sequential`; this gives the freeze policy and summary a stable way to identify both pieces. An explicit feature-forward path was chosen so the wrapper can keep the classifier separate while retaining the standard ResNet stem and stages.
## Traps: why the obvious fix is wrong
- `torchvision` weight constructors may fetch missing checkpoints automatically; checking the torch hub cache first keeps model construction offline and points users to the owner-run fetch script.
- Calling `.train()` recursively re-enables BatchNorm in earlier frozen stages. The wrapper reapplies eval mode to those layers, while leaving layer4 BatchNorm trainable in `last_block` mode.
- In `last_block`, the final ResNet `fc` is removed from the feature extractor. Its parameters must not be counted as trainable or mistaken for the custom head.
## Components
- `src/ie4483/model.py::build_model` — construction, cache check, head, freeze policy
- `src/ie4483/model.py::model_summary` — shape and parameter report
- `src/ie4483/model.py::_ResNetWithHead` — forward and BatchNorm mode handling
- `tests/test_model.py` — CPU behavior coverage
## Breaks or needs follow-up
None. The existing `freeze_backbone` value maps true to `frozen` and false to `full`; callers needing `last_block` or custom head values must currently pass those options through a model mapping/object. Add typed config fields in a later scoped config issue if YAML-driven selection is needed.
## Problem and constraints
The transfer-learning pipeline needs both requested ImageNet ResNet variants, a project-owned classifier, selective fine-tuning, and useful model size/shape information. Tests must not download weights, use real images, or require a GPU. Changes are limited to the issue's model and base-config scope, with the required tests and decision record included.
## Alternatives considered
- Replace `fc` with a simple single linear layer only — insufficient for optional dropout and hidden-width experiments.
- Use `requires_grad` alone for frozen mode — insufficient because frozen BatchNorm running statistics would still change in training mode.
## Trade-offs
Keeping optional model settings outside the typed YAML schema avoids a scope expansion but means they are not validated during `load_config` and are not documented as base YAML keys. The summary performs a real forward pass, so its cost scales with image size and backbone; it avoids an extra dependency and reports the actual output shape.
## Key parameters
- `num_classes: 2` (`src/ie4483/model.py::build_model`) — default Dogs vs Cats head; increasing it changes the output width and final layer size.
- `dropout: 0.0` (`src/ie4483/model.py::build_model`) — disabled by default; increasing it applies stronger stochastic regularization in training.
- `hidden_dim: None` (`src/ie4483/model.py::build_model`) — direct linear head by default; a positive width adds a learned hidden layer.
- `freeze_mode: frozen` when unspecified (`src/ie4483/model.py::build_model`) — keeps the existing boolean-config behavior; choose `last_block` or `full` to expose more backbone parameters to updates.
- Feature dimensions 512 and 2048 (`src/ie4483/model.py::_BACKBONES`) — torchvision ResNet-18 and ResNet-50 pooled output widths; changing backbone requires matching this dimension.
## Assumptions
Torchvision's ResNet weight enums and model signatures match the installed torchvision, and its selected default checkpoints use the URLs embedded in those enums. The factory supports the existing `Config` object and dictionaries or objects with equivalent fields; arbitrary unrelated config shapes are not normalized.
## Edge cases
- Handled: unsupported architectures, invalid class/dropout/hidden sizes, invalid freeze modes, missing cached pretrained checkpoints, and non-positive summary image sizes produce errors.
- Handled: summary restores the model's prior train/eval state, and frozen BatchNorm remains in eval after `.train()`.
- Not handled: a pretrained checkpoint that exists but is corrupt or incompatible is reported by torchvision's state loader.
## Not verified
No pretrained weights were loaded because this issue's tests must use `pretrained: false`; the successful cached-weight loading path was not exercised.
## Changes outside the scope
`tests/test_model.py` and this required decision record are outside the literal `src/ie4483/model.py`, `configs/base.yaml` scope; both are necessary to meet the repository's test and decision-record rules. `configs/base.yaml` and `README.md` were not changed because no config key, command, or script was added.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_model.py -q` passed 12 tests. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed all 30 tests in 49.12 seconds on CPU. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check --fix src/ie4483/model.py tests/test_model.py` fixed import ordering, then `ruff check` on those files passed.
## To change this
- To make last-block fine-tuning the default, set `freeze_mode` to `last_block` in the model mapping; this enables layer4 updates and its BatchNorm statistics.
- To add a hidden head, pass a positive `hidden_dim`; this adds parameters and activation memory.
- To change which pretrained checkpoint is used, change the weight enum in `src/ie4483/model.py::_BACKBONES` and keep `scripts/fetch_weights.py` cache behavior aligned.
## Rollback
Remove `src/ie4483/model.py` and `tests/test_model.py`, and revert this record. No existing module imports the factory yet. Before rollback, check for downstream training code that began importing `ie4483.model` after this change.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:26 UTC for issue #6 attempt 6; compared HEAD..working tree (base 8631003). Records are claims to check against the code, not proof.

**Change summary**

- `src/ie4483/`: 1 file(s), 0 component(s), +143/-0 lines
- plus 1 test file(s)
- new files: `src/ie4483/model.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/model.py`, `configs/base.yaml`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

3 checked by the tool, 0 tested in this run, 16 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 3 checked, 0 tested, 1 claimed
- Key parameters: 0 checked, 0 tested, 5 claimed
- Edge cases: 0 checked, 0 tested, 2 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 3 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/model.py::build_model` — construction, cache check, head, freeze policy (named code exists and changed)
  - [checked] Components: `src/ie4483/model.py::model_summary` — shape and parameter report (named code exists and changed)
  - [checked] Components: `src/ie4483/model.py::_ResNetWithHead` — forward and BatchNorm mode handling (named code exists and changed)
  - [claimed] Components: `tests/test_model.py` — CPU behavior coverage (names no code)
  - [claimed] Key parameters: `num_classes: 2` (`src/ie4483/model.py::build_model`) — default Dogs vs Cats head; incr... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: unsupported architectures, invalid class/dropout/hidden sizes, invalid freeze... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_model.py -q` passe... (no test named)
  - [claimed] Key parameters: `dropout: 0.0` (`src/ie4483/model.py::build_model`) — disabled by default; increasing i... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: summary restores the model's prior train/eval state, and frozen BatchNorm rema... (names no code)
  - [claimed] Key parameters: `hidden_dim: None` (`src/ie4483/model.py::build_model`) — direct linear head by default... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 0 warning(s) between the rationale above and the change.

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0006-resnet-model-factory.md` gives the commit.
<!-- verified-facts:end -->
