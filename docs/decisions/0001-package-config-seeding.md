# 0001: Package skeleton, config loading and seeding
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #1 · Date: 2026-10-19
## In short
The repository now has an installable `ie4483` source package, YAML-backed typed configuration, and a shared deterministic seed helper. Base settings and a test suite establish the pipeline defaults without accessing images, weights, or a GPU.
## How it works
1. A caller loads `configs/base.yaml`, or another YAML file, as the root of a configuration tree. The loader creates separate data, model, training, and evaluation dataclasses, retaining defaults for any fields the file omits.
2. Each supplied section and field is checked against that tree. Dotted `--set` expressions are parsed as YAML values, so `false`, integers, and lists keep their types. Unknown keys, malformed expressions, and type mismatches fail during loading instead of silently running with a default.
3. The data root is selected from an explicit config value, then `IE4483_DATA`, then `data/`. Before a run begins, its caller can save the resolved values beside run outputs as `config.yaml`, and reload that file to recover the same settings.
4. Before stochastic work, the caller passes its seed to the shared helper. It seeds Python, NumPy, torch CPU, and available CUDA generators, and enables deterministic cuDNN behavior.
5. Tests exercise this contract with generated values on CPU. They do not require a training dataset or pretrained weights, so these setup checks remain usable on machines with no available GPU.
## Main design choice
The config uses standard-library dataclasses and PyYAML's safe parser rather than introducing a validation framework. This keeps the initial package small and readable; a schema library would be better if the config grows complex cross-field constraints or needs generated schemas.
## Traps: why the obvious fix is wrong
- Treating override values as strings makes `false`, integers, and lists silently behave incorrectly; YAML parsing preserves their intended types.
- Silently ignoring unknown keys hides spelling errors and means a requested experiment may run with defaults; loading therefore rejects them.
- Seeding only torch misses Python and NumPy randomness used in data and evaluation helpers.
- Always using a configured default data path would ignore a user's environment override; precedence is explicit config, environment, then repository-relative default.
## Components
- `pyproject.toml::project` — hatchling src package
- `src/ie4483/config.py::Config` — typed config and YAML helpers
- `src/ie4483/seed.py::seed_everything` — random generator setup
- `configs/base.yaml` — project defaults
- `tests/test_config.py` — loading and overrides
- `tests/test_seed.py` — reproducibility
## Breaks or needs follow-up
None.
## Problem and constraints
This is pipeline groundwork for an image classification project. It must not download data or weights, and must run on CPU. Defaults describe the Dogs vs Cats phase and config resolution must be reproducible and inspectable.
## Alternatives considered
- Pydantic or another schema package — useful for richer validation, but unnecessary dependency and machinery for four straightforward sections.
- String-only override handling — simpler parser, but breaks values such as numeric seeds and booleans.
## Trade-offs
Dataclass checking catches unknown keys and basic scalar type mismatches but does not enforce every semantic relationship between settings. Add focused checks as later pipeline requirements become concrete.
## Key parameters
- `seed: 42` (`configs/base.yaml::train.seed`) — reproducible default; changing it selects another random stream.
- `image_size: 224` (`configs/base.yaml::data.image_size`) — standard ResNet input size; changing it alters preprocessing and compute.
- `batch_size: 32` (`configs/base.yaml::train.batch_size`) — initial project default; later hardware tuning may lower it.
## Assumptions
PyYAML, NumPy, and torch are available in the project's shared environment. If config is copied away from the repo and has no explicit data root or environment override, its fallback `data/` is relative to the current working directory.
## Edge cases
- Handled: unknown keys, malformed `key=value`, non-mapping YAML sections, and no CUDA device.
- Not handled: cross-field validation beyond section field types; later pipeline issues should add it when requirements are known.
## Not verified
No training or model inference was run; this issue adds setup only.
## Changes outside the scope
None.
## How it was verified
`tests/test_config.py` covers config loading, dumping and reloading, typed overrides, unknown-key rejection, and data-root precedence. `tests/test_seed.py` compares Python, NumPy, and torch outputs after reseeding. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed all 6 tests; `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check src tests` reported all checks passed.
## To change this
To add a setting, add its dataclass field and matching `configs/base.yaml` entry; the loader rejects config keys without a field. To change the default data root, update `data.root` in the base config.
## Rollback
Remove the package, config, and tests added for issue #1 and restore the prior README and ignore rules; no existing caller depends on these new modules.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:02 UTC for issue #1 attempt 1; compared HEAD..working tree (base 6c0cf86). Records are claims to check against the code, not proof.

**Change summary**

- `(root)/`: 2 file(s), 0 component(s), +21/-2 lines
- `configs/`: 1 file(s), 0 component(s), +18/-0 lines
- `src/ie4483/`: 3 file(s), 0 component(s), +165/-0 lines
- plus 2 test file(s), 1 doc file(s)
- new files: `configs/base.yaml`, `pyproject.toml`, `src/ie4483/__init__.py`, `src/ie4483/config.py`, `src/ie4483/seed.py`

**Removed symbols**

None.

**Scope**

Declared: `pyproject.toml`, `src/ie4483/__init__.py`, `src/ie4483/seed.py`, `src/ie4483/config.py`, `configs/**`, `tests/**`, `README.md`, `.gitignore`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 1 tested in this run, 13 claimed only (of 18 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `pyproject.toml::project` — hatchling src package (named code exists and changed)
  - [tested] How it was verified: `tests/test_config.py` covers config loading, dumping and reloading, typed overrides, u... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/config.py::Config` — typed config and YAML helpers (named code exists and changed)
  - [checked] Components: `src/ie4483/seed.py::seed_everything` — random generator setup (named code exists and changed)
  - [checked] Components: `configs/base.yaml` — project defaults (named code exists and changed)
  - [claimed] Components: `tests/test_config.py` — loading and overrides (names no code)
  - [claimed] Key parameters: `seed: 42` (`configs/base.yaml::train.seed`) — reproducible default; changing it select... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: unknown keys, malformed `key=value`, non-mapping YAML sections, and no CUDA de... (names no code)
  - [claimed] Components: `tests/test_seed.py` — reproducibility (names no code)
  - [claimed] Key parameters: `image_size: 224` (`configs/base.yaml::data.image_size`) — standard ResNet input size;... (no cited constant with a matching value)
  - full per-claim list (18) in the JSON record

**Consistency**

0 error(s), 4 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 6 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: references something that doesn't exist: config.yaml
- WARN: .gitignore (modified) changed but not mentioned in the record
- WARN: src/ie4483/__init__.py (new file) changed but not mentioned in the record

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0001-package-config-seeding.md` gives the commit.
<!-- verified-facts:end -->
