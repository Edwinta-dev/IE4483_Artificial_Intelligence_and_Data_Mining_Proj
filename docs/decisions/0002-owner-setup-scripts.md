# 0002: Owner setup scripts: cache ResNet weights and check the data layout
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #2 · Date: 2026-10-19
## In short
The repository now has a local setup report and an owner-run command that caches the default torchvision ResNet-18 and ResNet-50 ImageNet weights. Data-root resolution is shared with the existing config, and README instructions describe the expected data layout and commands.
## How it works
1. Place the Dogs vs Cats data under the default `data/` directory or select another root through the YAML config or `IE4483_DATA`. The configured root takes priority, then the environment variable, then the default.
2. Run the fetch command once while online. Torchvision downloads the two default ResNet weight sets to its hub checkpoints directory, which follows `TORCH_HOME` when set.
3. Run the setup check locally. It prints Python and PyTorch versions, attempts a small CUDA tensor allocation when CUDA reports available, and includes any allocation error instead of stopping.
4. The report checks both expected checkpoint filenames and lists image counts for the training, validation, and test folders. Missing folders are marked, and counts include recognized image file suffixes only.
5. Tests create temporary data folders and cache files. They inspect report contents without invoking the fetch command or network access.
## Main design choice
Torchvision's weight enums and state-dictionary loader provide the official checkpoint names, hash checks, and torch hub cache behavior. The alternative was manually downloading URL strings and choosing cache filenames; that duplicates torchvision metadata and risks drift. A second alternative was embedding setup checks in the training runner; standalone owner commands keep network access and machine diagnosis outside the training/test path.
## Traps: why the obvious fix is wrong
- A CUDA availability flag does not prove that the installed driver can allocate a tensor; the report performs an allocation and captures its failure.
- Missing folders should remain visible rather than making setup inspection crash; the report emits zero and marks the directory missing.
- Image counts should not include notes or metadata files; only the known image suffixes are counted.
- Fetching weights at import time would make tests or inspection unexpectedly use the network; downloads happen only when the fetch script is run as a command.
## Components
- `src/ie4483/paths.py::resolve_data_root` — shared data precedence
- `src/ie4483/paths.py::runs_dir` — repository run directory
- `src/ie4483/paths.py::weights_cache_dir` — torch hub checkpoints
- `src/ie4483/config.py::Config.data_root` — config integration
- `scripts/fetch_weights.py::main` — owner weight fetch
- `scripts/check_setup.py::inspect_setup` — local diagnostic report
- `tests/test_setup.py` — generated layout and cache reports
- `README.md` — setup instructions
## Breaks or needs follow-up
None.
## Problem and constraints
Owner setup needs to prepare cached weights and detect missing local data before pipeline work begins. Tests and normal setup inspection must not reach the network. The project also needs to work when CUDA is absent or its driver cannot allocate a tensor.
## Alternatives considered
- Handwritten checkpoint URLs and filenames — not chosen because torchvision already owns those details.
- One combined setup command that downloads automatically — not chosen because diagnostics should be safe to run offline and in tests.
## Trade-offs
The checker reports counts and presence, not whether images can be decoded or whether labels are correct. That keeps setup quick and makes no claim about dataset quality; validation belongs in later data-pipeline checks.
## Key parameters
- `WEIGHT_FILES` (`scripts/check_setup.py::WEIGHT_FILES`): `resnet18-f37072fd.pth` and `resnet50-11ad3fa6.pth`, matching torchvision's current default enum checkpoints; changing architecture or weights requires updating this list alongside the fetch command.
- `IMAGE_SUFFIXES` (`scripts/check_setup.py::IMAGE_SUFFIXES`): common raster image suffixes; adding a format changes which files contribute to folder counts.
- `data/` (`src/ie4483/paths.py::resolve_data_root`): fallback relative to the current working directory, consistent with the existing configuration behavior.
## Assumptions
Torchvision's default ResNet-18 and ResNet-50 weight enums continue to use these checkpoint filenames. The project data layout uses lowercase `cat` and `dog` directory names, while image extensions may vary in case.
## Edge cases
- Handled: missing data directories, non-image files, no CUDA device, CUDA allocation errors, absent checkpoints, and a custom `TORCH_HOME`.
- Not handled: unreadable/corrupt images, empty-but-existing class folders as an error, alternate label directory spellings, and checking checkpoint integrity after caching.
## Not verified
The actual fetch command was not run because the task prohibits network use during this work. CUDA allocation availability on the host and real dataset contents were not asserted by these tests.
## Changes outside the scope
None.
## How it was verified
`tests/test_setup.py` checks complete and incomplete generated data layouts, image-only counts, and present/missing weight reports. The full CPU suite passed: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` reported 8 passed. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check src scripts tests` reported all checks passed.
## To change this
To use a different torchvision weight variant, change the enum selected in `scripts/fetch_weights.py` and keep `scripts/check_setup.py::WEIGHT_FILES` aligned with its cached filename. To recognize another image format, add its lowercase suffix to `scripts/check_setup.py::IMAGE_SUFFIXES`.
## Rollback
Remove the setup scripts, shared paths module, setup tests, README setup section, and this record. Restore `Config.data_root` to its prior inline precedence logic; no training caller has been added by this issue.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:05 UTC for issue #2 attempt 2; compared HEAD..working tree (base 5237a53). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 2 file(s), 0 component(s), +84/-0 lines
- `src/ie4483/`: 2 file(s), 1 component(s), +28/-2 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/check_setup.py`, `scripts/fetch_weights.py`, `src/ie4483/paths.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/fetch_weights.py`, `scripts/check_setup.py`, `src/ie4483/paths.py`, `README.md`. Verdict: violation.
- out of scope: `src/ie4483/config.py::Config.data_root` (modified); explained in the record: no

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

6 checked by the tool, 1 tested in this run, 13 claimed only (of 20 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 6 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/paths.py::resolve_data_root` — shared data precedence (named code exists and changed)
  - [tested] How it was verified: `tests/test_setup.py` checks complete and incomplete generated data layouts, image-only... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/paths.py::runs_dir` — repository run directory (named code exists and changed)
  - [checked] Components: `src/ie4483/paths.py::weights_cache_dir` — torch hub checkpoints (named code exists and changed)
  - [checked] Components: `src/ie4483/config.py::Config.data_root` — config integration (named code exists and changed)
  - [checked] Components: `scripts/fetch_weights.py::main` — owner weight fetch (named code exists and changed)
  - [checked] Components: `scripts/check_setup.py::inspect_setup` — local diagnostic report (named code exists and changed)
  - [claimed] Components: `tests/test_setup.py` — generated layout and cache reports (names no code)
  - [claimed] Key parameters: `WEIGHT_FILES` (`scripts/check_setup.py::WEIGHT_FILES`): `resnet18-f37072fd.pth` and `r... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing data directories, non-image files, no CUDA device, CUDA allocation err... (names no code)
  - full per-claim list (20) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 3 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: out-of-scope change not explained in 'Changes outside the scope': src/ie4483/config.py::Config.data_root

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0002-owner-setup-scripts.md` gives the commit.
<!-- verified-facts:end -->
