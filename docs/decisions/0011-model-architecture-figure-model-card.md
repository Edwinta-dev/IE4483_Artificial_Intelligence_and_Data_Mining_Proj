# 0011: Model architecture figure and model card for the report
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #11 · Date: 2026-10-19
## In short
The repository now generates a report diagram as PNG and PDF and a Markdown model card from a model config. Both artifacts describe the instantiated ResNet and use its model summary for input/output shapes and parameter counts.
## How it works
1. The command reads the YAML file, including the training settings that the training pipeline supports in addition to the strict typed config, and seeds the project RNGs.
2. It builds the configured ResNet-18 or ResNet-50. The normal factory behavior applies, including offline-only pretrained weight loading and the configured frozen, last-block, or full fine-tuning policy.
3. The figure helper summarizes a zero-valued input at the configured image size and draws the input, stem, four ResNet stages, pooled feature vector, custom head layers, and class scores. Each block shows its channel and spatial dimensions where applicable and whether its parameters can train.
4. It saves `model_architecture.png` and `model_architecture.pdf` to the chosen output folder. The card helper writes `model_card.md` in the same folder with the backbone and pretrained source, head, loss, optimizer, schedule, freeze mode, dimensions, and parameter totals.
5. Users point the output argument at a run folder so generated report artifacts stay with that run rather than in the source tree.
## Main design choice
The diagram is constructed from the existing model wrapper and its known ResNet stage strides, while the card obtains its tensor dimensions and parameter counts from `src/ie4483/model.py::model_summary`. A second candidate was to register torchvision forward hooks and execute every stage to discover shapes. That would track arbitrary backbone substitutions automatically, but this project only supports the two known ResNets, and formula-derived stage dimensions avoid another image pass and hook cleanup. The card uses the actual instantiated head modules rather than assuming a fixed linear layer, so optional dropout and hidden layers remain visible. The output folder is explicit; silently defaulting report outputs into the repository risks leaving generated figures in source control.
## Traps: why the obvious fix is wrong
- A `pretrained: true` setting does not prove the report model loaded weights: the factory requires the checkpoint to exist in the local torch cache and deliberately performs no download. The card reports the configured torchvision source; its source line is not evidence of successful weight acquisition.
- `freeze_backbone` and `freeze_mode` both affect training policy. The existing factory resolves the legacy boolean into a mode, and diagram labels use the resulting model mode rather than reading the YAML key directly.
- A model card must describe the same head and parameter totals as the instantiated model. Hard-coding the expected direct linear head would misstate optional hidden and dropout layers.
- The typed config rejects extra train settings. The command uses the existing training-config loader to preserve the optimizer and schedule conventions used by training.
## Components
- `scripts/model_figure.py::main` — command and output path
- `src/ie4483/figures.py::model_architecture_figure` — PNG/PDF diagram
- `src/ie4483/figures.py::write_model_card` — Markdown card
- `src/ie4483/model.py::model_summary` — dimensions and parameter counts
- `tests/test_figures.py::test_report_model_artifacts_match_forward_shapes` — both backbones
## Breaks or needs follow-up
None. The command is additive and does not alter model construction or training.
## Problem and constraints
Rubric part (b) requires a legible architecture figure with input/output dimensions and a model card grounded in the configured model. The work stays offline and supports CPU-only model construction with `pretrained: false`; it never trains or consumes project data. Report outputs belong in a run folder, and tests exercise the two supported backbones with small synthetic tensors.
## Alternatives considered
- Torchvision hooks for all intermediate shapes — not chosen because only the fixed ResNet-18/50 stage layout is supported and shape calculation avoids another forward pass.
- Store generated files beside the script — rejected because it would mix run-specific report artifacts into the repository.
## Trade-offs
The stage shape calculation follows the torchvision ResNet stem and stage downsampling layout, so a future backbone with different strides would need this renderer updated. The explicit output argument makes invocation slightly longer but prevents accidental artifact placement. The card describes configured training settings, not completed training results or measured performance.
## Key parameters
- `data.image_size` (`src/ie4483/config.py::DataConfig`) — input height and width used for the diagram and summary; larger values change spatial stage shapes.
- `model.architecture` (`src/ie4483/config.py::ModelConfig`) — selects ResNet-18 or ResNet-50 and therefore the stage feature widths.
- `train.epochs` (`src/ie4483/config.py::TrainConfig`) — cosine schedule horizon shown in the card; changing it changes the described schedule.
- Output directory (`scripts/model_figure.py::main --output`) — destination for all three artifacts.
## Assumptions
The instantiated backbone follows the standard torchvision ResNet stem and stage strides, and `model_summary` can run one zero tensor on the model's current device. The configured pretrained flag and torchvision's default weight enum identify the intended source; a cached checkpoint is still needed if pretrained construction is requested.
## Edge cases
- Handled: absent output directories are created; invalid input sizes fail through `model_summary`; unrecognized config/model settings retain errors from existing config and model loaders.
- Not handled: custom backbone implementations with different stage strides would receive incorrect intermediate spatial labels.
## Not verified
- The default base config requests pretrained weights; the command was not run against it because this environment is not guaranteed to contain the local checkpoints. Tests use `pretrained: false` as required.
- No real training run or accuracy claim is represented by the card.
## Changes outside the scope
`README.md` documents the added command as required when a script is introduced. `tests/test_figures.py` and this decision record are required supporting files; no other implementation files were changed.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_figures.py -q` passed 2 tests. The tests instantiate both backbones without pretrained weights, compare card shapes against an actual forward pass and `model_summary`, and check that PNG/PDF artifacts were written. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check scripts/model_figure.py src/ie4483/figures.py tests/test_figures.py` passed. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed all 45 tests on CPU in 71.66 seconds.
## To change this
- To show a different report input size, change `data.image_size`; it affects the model's real input tensor and all reported stage dimensions.
- To add a supported backbone, update `src/ie4483/model.py::_BACKBONES` and the stage channel/stride rendering in `src/ie4483/figures.py::model_architecture_figure` together; otherwise figure dimensions can disagree with the backbone.
- To change artifact names or add formats, update the save paths in `src/ie4483/figures.py::model_architecture_figure` and README command description.
## Rollback
Remove `scripts/model_figure.py` and `src/ie4483/figures.py`, revert the README command and remove `tests/test_figures.py`. The model factory and training pipeline do not depend on these report helpers.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 03:05 UTC for issue #11 attempt 11; compared HEAD..working tree (base 9858d53). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 0 component(s), +31/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +99/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/model_figure.py`, `src/ie4483/figures.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/model_figure.py`, `src/ie4483/figures.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

3 checked by the tool, 0 tested in this run, 17 claimed only (of 20 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 3 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 2 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `scripts/model_figure.py::main` — command and output path (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::model_architecture_figure` — PNG/PDF diagram (named code exists and changed)
  - [checked] Components: `src/ie4483/figures.py::write_model_card` — Markdown card (named code exists and changed)
  - [claimed] Components: `src/ie4483/model.py::model_summary` — dimensions and parameter counts (`src/ie4483/model.py::model_summary` exists but did not change)
  - [claimed] Key parameters: `data.image_size` (`src/ie4483/config.py::DataConfig`) — input height and width used fo... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: absent output directories are created; invalid input sizes fail through `model... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest tests/test_figures.py -q` pas... (no test named)
  - [claimed] Components: `tests/test_figures.py::test_report_model_artifacts_match_forward_shapes` — both backbones (names no code)
  - [claimed] Key parameters: `model.architecture` (`src/ie4483/config.py::ModelConfig`) — selects ResNet-18 or ResNe... (no cited constant with a matching value)
  - [claimed] Key parameters: `train.epochs` (`src/ie4483/config.py::TrainConfig`) — cosine schedule horizon shown in... (no cited constant with a matching value)
  - full per-claim list (20) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 3 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: references something that doesn't exist: model_card.md

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0011-model-architecture-figure-model-card.md` gives the commit.
<!-- verified-facts:end -->
