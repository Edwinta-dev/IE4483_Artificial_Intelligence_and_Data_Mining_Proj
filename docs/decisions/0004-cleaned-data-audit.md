# 0004: Cleaned-data handoff audit
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #4 · Date: 2026-10-19
## In short
A handoff audit now reports the expected Dogs vs Cats directory layout, image counts, unreadable files, image dimensions, exact train duplicates, train-to-validation/test duplicates, and file-list differences against another root. It writes `audit.json` and `audit.md` in the audited data root; the command exits unsuccessfully for a missing required directory or train/validation leakage.
## How it works
1. The user points the audit command at the cleaned dataset root and may also give it another root, such as the original data tree.
2. The audit checks for `train/cat`, `train/dog`, `val/cat`, `val/dog`, and `test`. It lists supported image files in each location and records counts.
3. Each image is decoded to check readability and to collect its width and height. For train files, it calculates SHA-256 hashes and groups identical files. It compares those hashes with validation and test images to flag leakage.
4. If a comparison root is supplied, it compares relative image paths and lists paths present only in either tree.
5. It saves a machine-readable JSON report and a readable Markdown report under the audited root, and prints the Markdown report. A missing required directory or any train/validation duplicate makes the command return status 1.
## Main design choice
The audit is implemented as a small standalone module that can be called directly and through a thin script. A plausible alternative was to extend `scripts/check_setup.py`; that would put this handoff-specific report alongside setup and weight-cache checks, while the separate command provides the requested `audit_data.py` entry point. A second alternative was to make the audit only a script; the reusable module won because tests can inspect a structured report without parsing console output.
## Traps: why the obvious fix is wrong
- File counts alone conceal corrupt image files; the audit decodes images and records failures separately.
- Comparing filenames for duplicates misses copies renamed during cleaning; SHA-256 compares file contents.
- Comparing train only with validation would miss overlap with the unlabelled test set, so the report keeps separate validation and test leakage lists.
- A train/test overlap is reported, but it does not determine the command's exit status; the issue specifically gates that status on layout and train/validation leakage.
- `audit.json` and `audit.md` are generated inside the audited root. The comparison listing filters by image suffix, so those reports do not appear as dataset files.
## Components
- `src/ie4483/data_audit.py::audit_data` — report generation
- `src/ie4483/data_audit.py::EXPECTED_DIRS` — required layout
- `src/ie4483/data_audit.py::IMAGE_SUFFIXES` — recognized formats
- `scripts/audit_data.py::main` — command and exit status
- `tests/test_data_audit.py` — generated-tree acceptance coverage
- `README.md` — usage examples
## Breaks or needs follow-up
None.
## Problem and constraints
The team needs to inspect cleaned data as it arrives, before training begins. This is groundwork only: it uses tiny generated images in tests and does not train, download data, load weights, or require a GPU. The existing loader recognizes raster suffixes, and this audit uses the same set.
## Alternatives considered
- Add audit options to `scripts/check_setup.py` — rejected because that command reports runtime and weight readiness, while the new command is specifically a data handoff audit.
- Emit only console text — rejected because JSON and Markdown artifacts are explicit outputs and JSON is useful for later tooling.
- Compare content hashes across the whole tree — the implemented leakage comparison is scoped to train versus validation/test, while `--compare` compares relative file listings as requested.
## Trade-offs
The audit reads every discovered image and hashes train and evaluation/test files, which adds disk I/O proportional to the dataset size. This gives an exact content check without relying on filenames; caching could reduce repeat cost for larger roots but would risk stale results after cleaning changes.
## Key parameters
- `IMAGE_SUFFIXES` (`src/ie4483/data_audit.py::IMAGE_SUFFIXES`) — formats included in counts, dimensions, hashing, and comparison; adding a suffix includes more files.
- `EXPECTED_DIRS` (`src/ie4483/data_audit.py::EXPECTED_DIRS`) — directory names required for layout success; changing it changes the reported layout contract.
- Hash chunk size 1 MiB (`src/ie4483/data_audit.py::_hash`) — bounds per-read memory while hashing; changing it affects I/O granularity, not digest results.
## Assumptions
A duplicate means byte-identical file content, which SHA-256 detects for practical handoff purposes. Relative paths are the right comparison unit for raw-versus-clean listings; renamed files will appear as one removal and one addition. Image files use one of the configured raster suffixes.
## Edge cases
- Handled: absent required directories; empty folders; unreadable image files; no readable images for dimension statistics; even-sized medians; duplicate files with different names; missing optional comparison roots; files present only on either side of a comparison.
- Not handled: perceptually identical images re-encoded with different bytes; they will not be reported as exact duplicates.
## Not verified
No real dataset was audited. The report's behavior on very large datasets and file permission changes during scanning was not measured.
## Changes outside the scope
- `README.md`: documented the new command as required for added scripts.
- `tests/test_data_audit.py`: added generated-tree coverage required by project instructions.
- `docs/decisions/0004-cleaned-data-audit.md`: required issue decision record.
## How it was verified
`C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_data_audit.py` passed the generated clean, missing-folder/corrupt-file, in-train duplicate, train/validation leakage exit-code, and compare-listing cases. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check src/ie4483/data_audit.py scripts/audit_data.py tests/test_data_audit.py` passed. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed the full CPU suite.
## To change this
To recognize an image suffix, update `src/ie4483/data_audit.py::IMAGE_SUFFIXES` and keep it aligned with `src/ie4483/data.py::IMAGE_SUFFIXES`. To require another folder, update `src/ie4483/data_audit.py::EXPECTED_DIRS`. To change when the command fails, update the condition in `scripts/audit_data.py::main` and its tests.
## Rollback
Remove `src/ie4483/data_audit.py`, `scripts/audit_data.py`, `tests/test_data_audit.py`, and this record, then remove the README usage paragraph. No existing pipeline component depends on the audit.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:12 UTC for issue #4 attempt 4; compared HEAD..working tree (base 95ab7d6). Records are claims to check against the code, not proof.

**Change summary**

- `scripts/`: 1 file(s), 0 component(s), +27/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +115/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/audit_data.py`, `src/ie4483/data_audit.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/data_audit.py`, `scripts/audit_data.py`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 0 tested in this run, 15 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 3 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 0 tested, 1 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/data_audit.py::audit_data` — report generation (named code exists and changed)
  - [checked] Components: `src/ie4483/data_audit.py::EXPECTED_DIRS` — required layout (named code exists and changed)
  - [checked] Components: `src/ie4483/data_audit.py::IMAGE_SUFFIXES` — recognized formats (named code exists and changed)
  - [checked] Components: `scripts/audit_data.py::main` — command and exit status (named code exists and changed)
  - [claimed] Components: `tests/test_data_audit.py` — generated-tree acceptance coverage (names no code)
  - [claimed] Key parameters: `IMAGE_SUFFIXES` (`src/ie4483/data_audit.py::IMAGE_SUFFIXES`) — formats included in cou... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: absent required directories; empty folders; unreadable image files; no readabl... (names no code)
  - [claimed] How it was verified: `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q tests/test_data_audit.py`... (no test named)
  - [claimed] Components: `README.md` — usage examples (names no code)
  - [claimed] Key parameters: `EXPECTED_DIRS` (`src/ie4483/data_audit.py::EXPECTED_DIRS`) — directory names required... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 3 warning(s) between the rationale above and the change.
- WARN: 'How it works' has 5 code references (path::Name or `identifiers`); it is for human readers - describe the steps in plain language, put names in Components
- WARN: references something that doesn't exist: audit.json
- WARN: references something that doesn't exist: audit.md

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0004-cleaned-data-audit.md` gives the commit.
<!-- verified-facts:end -->
