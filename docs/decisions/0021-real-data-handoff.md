# 0021: Real data handoff preprocessing and audit
<!-- verify-banner:begin -->
> WARNING: Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool | [tested] cited test exists and passed in this run | [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #21 · Date: 2026-10-26
## In short
The team's source handoff is preprocessed into the ignored `data/` tree and passes the integrity audit. Preprocessing retains one converted copy of byte-identical outputs, preferring train, then validation, then test, so duplicate copies cannot leak across those splits.
## How it works
1. The owner places the supplied folders under `raw-data/datasets/` and runs the preprocessing command from the repository root.
2. The converter finds supported images in a stable order, with training files ahead of validation and test files. It clears supported image files in the generated destination so an earlier run cannot leave stale data behind.
3. Each source image is oriented using its EXIF metadata, converted to RGB, resized to a square with a centre-fit crop, and saved as a JPEG. The output bytes are hashed; when those bytes already exist in an earlier file, the duplicate output is removed.
4. The audit checks folder layout, decodability, dimensions, within-train duplicates, and overlap from training into validation and test. It saves JSON and Markdown reports in `data/`.
5. The handoff note records the counts from that audit and explains the preprocessing crop alongside the runtime evaluation and training transforms.
## Main design choice
Duplicates are removed during conversion using hashes of the actual output JPEG bytes, in deterministic train/validation/test priority order. Manually deleting source images would alter the team's handoff, while relaxing `scripts/audit_data.py::main` would hide leakage. Output de-duplication retains the earliest split occurrence and lets the existing audit remain an independent check. Hashing converted files also catches distinct source encodings that become identical after resize/encoding; it can theoretically miss perceptually identical images whose output bytes differ.
## Traps: why the obvious fix is wrong
- Fixing the audit's exit condition would make the command green while retaining leakage; the audit is meant to expose data problems.
- Deleting or renaming raw files would change the handoff and make source comparison harder to reproduce.
- Processing arbitrary filesystem order could let a validation copy win over its training counterpart. Split priority and relative-path sorting make retention repeatable.
- Rerunning without clearing prior generated images could preserve removed or renamed files in the destination and make audit counts misleading.
- Duplicate detection is on converted output bytes, so the checked condition matches what the audit sees.
## Components
- `src/ie4483/preprocess_images.py::preprocess_images` — conversion and duplicate filtering
- `scripts/audit_data.py::main` — audit command and exit status
- `src/ie4483/data_audit.py::audit_data` — integrity report
- `tests/test_preprocess_images.py` — generated conversion fixture
- `docs/data_handoff.md` — real counts and policy
## Breaks or needs follow-up
The generated output omits duplicate image files, so its class counts are lower than raw source counts by the number of duplicates in those classes. This is intentional to remove leakage; raw source data remains unchanged. No other known break.
## Problem and constraints
Issue #21 requires a real local handoff conversion and audit, no downloads or model training, and an audit-clean result. `raw-data/datasets/train` exists. Data outputs are generated under ignored `data/` and are not source-controlled artifacts. Automated tests use small generated images.
## Alternatives considered
- Keep all files and only document leakage — rejected because the requested audit must exit 0.
- Change the audit to ignore duplicates — rejected because it would conceal a real integrity problem and degrade its contract.
- Remove duplicate inputs manually — rejected because the issue requires fixing preprocessing and leaves the team's raw handoff intact.
## Trade-offs
Deduplication adds hashing and removes some nominal examples; exact copies add no image variety, and retaining only the training copy protects evaluation. This policy would need reconsideration if repeated images were intentional independent observations or metadata identity mattered downstream.
## Key parameters
- Output size `(224, 224)` (`src/ie4483/preprocess_images.py::image_size`) — required handoff resolution; larger sizes increase disk and decode cost.
- JPEG quality 95 (`src/ie4483/preprocess_images.py::preprocess_images`) — fixed conversion setting; lower quality saves space but adds more compression loss.
- Split priority train, validation, test (`src/ie4483/preprocess_images.py::split_order`) — earlier split retains duplicates; changing order changes which split owns shared images.
- Pillow LANCZOS (`src/ie4483/preprocess_images.py::preprocess_images`) — resampling filter for resize; changing the filter changes output pixels and potentially hashes.
## Assumptions
The output directory is generated data owned by this preprocessing command; it removes supported raster files there before rebuilding. Exact output-byte identity is the duplicate definition, matching `src/ie4483/data_audit.py::_hash`. If users place unrelated images in the output root, those supported files will be deleted on rerun.
## Edge cases
- Handled: supported image conversion failures are recorded; same-output duplicates are omitted; repeated runs remove stale supported images; train wins cross-split duplicates.
- Not handled: perceptual duplicates with differing JPEG bytes remain; unsupported source extensions are not converted; unrelated supported files in the output directory are removed.
## Not verified
No training, model scoring, GPU operation, or download was performed. The audit comparison is based on relative filenames, and the raw handoff is nested below `raw-data/datasets`, so its path-difference lists are expected to be large and are not a content equality check.
## Changes outside the scope
- `tests/test_preprocess_images.py`: added the generated image test required by project instructions.
## How it was verified
`src/ie4483/preprocess_images.py` ran on the local handoff: 25,500 files found, 25,494 unique outputs retained, and no conversion failures (six exact output duplicates were omitted). `scripts/audit_data.py --root data --compare raw-data` exited 0. `data/audit.json` reported 9,998 train cats, 9,997 train dogs, 2,499 validation cats, 2,500 validation dogs, and 500 test images; it found no unreadable images, no exact train duplicates, and no train/validation or train/test leakage. The focused preprocessing and audit tests passed (6 tests), Ruff passed, and the full CPU suite passed (77 tests in 143.33 seconds) with one scheduler-order warning in `tests/test_train.py::test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch`.
## To change this
To change which split retains duplicate output, update `split_order` in `src/ie4483/preprocess_images.py`; earlier entries win, so changing order can reintroduce evaluation leakage. To recognize a source type, update `supported_formats` and confirm the dataset loader/audit supports it. To keep output files outside this generated root, pass `--output` and ensure the selected destination is safe to rebuild.
## Rollback
Revert the output deduplication and stale-image cleanup in `src/ie4483/preprocess_images.py`, restore the README usage text, and remove this issue's handoff note and decision record. Re-run conversion and audit before using `data/`; the prior output included known duplicate and validation-leakage findings.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-26 12:37 UTC for issue #21 attempt 2; compared HEAD..working tree (base 175c5a6). Records are claims to check against the code, not proof.

**Change summary**

- `src/ie4483/`: 1 file(s), 1 component(s), +23/-5 lines
- plus 1 test file(s), 2 doc file(s)

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/preprocess_images.py`, `scripts/audit_data.py`, `docs/**`, `README.md`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

1 checked by the tool, 1 tested in this run, 18 claimed only (of 20 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 1 checked, 0 tested, 4 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 5 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
- Breaks or needs follow-up: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/preprocess_images.py::preprocess_images` — conversion and duplicate filtering (named code exists and changed)
  - [tested] How it was verified: `src/ie4483/preprocess_images.py` ran on the local handoff: 25,500 files found, 25,494... (test gate passed: project test suite (CPU))
  - [claimed] Components: `scripts/audit_data.py::main` — audit command and exit status (`scripts/audit_data.py::main` exists but did not change)
  - [claimed] Key parameters: Output size `(224, 224)` (`src/ie4483/preprocess_images.py::image_size`) — required han... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: supported image conversion failures are recorded; same-output duplicates are o... (names no code)
  - [claimed] Components: `src/ie4483/data_audit.py::audit_data` — integrity report (`src/ie4483/data_audit.py::audit_data` exists but did not change)
  - [claimed] Key parameters: JPEG quality 95 (`src/ie4483/preprocess_images.py::preprocess_images`) — fixed conversi... (no cited constant with a matching value)
  - [claimed] Components: `tests/test_preprocess_images.py` — generated conversion fixture (names no code)
  - [claimed] Key parameters: Split priority train, validation, test (`src/ie4483/preprocess_images.py::split_order`)... (no cited constant with a matching value)
  - [claimed] Components: `docs/data_handoff.md` — real counts and policy (names no code)
  - full per-claim list (20) in the JSON record

**Consistency**

0 error(s), 2 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won
- WARN: references something that doesn't exist: src/ie4483/preprocess_images.py::split_order (no such component in the old or new version)

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0021-real-data-handoff.md` gives the commit.
<!-- verified-facts:end -->
