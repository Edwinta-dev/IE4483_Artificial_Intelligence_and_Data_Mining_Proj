# 0024: Augmented data branch on disk
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #24 · Date: 2026-10-27
## In short
The repository now has a seeded command that creates either an adjusted-only training view or a mixed view with originals and adjusted copies. The validation and test trees are copied as files, and a CSV captures every sampled operation and its parameters.
## How it works
1. The command receives an input dataset folder, an output folder, a mode, and a seed. It checks that the input exists, the mode is supported, and the output is not the input or nested beneath it.
2. For each file directly inside the cat and dog training folders, it decodes an RGB image and uses the seeded random stream to choose one of four adjustments.
3. A flip changes orientation, rotation samples an angle within the stated range, a resized crop samples an area and aspect ratio before resizing to 224 by 224, and colour jitter samples brightness, contrast, and saturation factors.
4. In adjusted-only mode each result uses the source filename. In mixed mode the original is copied and the adjusted JPEG uses the source stem with `_aug.jpg` appended.
5. The validation and test trees are copied without image decoding or recompression. A CSV row records each adjusted output and its operation and sampled values. The command prints counts for all four operation types.
## Main design choice
Materialize each adjusted image once with Pillow, keeping the training folder fixed so existing folder-based training and frozen-feature caching can be used. A runtime training transform would vary the image each epoch, but that prevents the current cache path from reusing frozen backbone features. Reusing the existing torchvision training transform would also couple this data-preparation command to its multi-operation configuration, while this issue requires exactly one randomly selected adjustment and an audit row per image. Pillow keeps the new helper self-contained and writes directly in the existing image-folder layout.
## Traps: why the obvious fix is wrong
- Applying multiple transforms to each image violates the one-adjustment-per-image contract and makes the CSV ambiguous.
- Resetting the random seed for each image would repeat the same operation and parameter sequence; the command initializes the shared seed once and advances a local stream through the dataset.
- Opening and saving validation or test images would alter JPEG bytes and break the held-out copy requirement; these trees are copied at file level.
- Mixed mode must retain the originals and use distinct `_aug.jpg` filenames or it will overwrite the source training files.
## Components
- `src/ie4483/augmented_view.py::make_augmented_view` — dataset writer
- `src/ie4483/augmented_view.py::_adjust` — one adjustment
- `scripts/make_augmented_view.py::main` — CLI
- `tests/test_augmented_view.py` — deterministic and layout coverage
- `README.md — usage
- `.gitignore` — generated data folders
## Breaks or needs follow-up
None.
## Problem and constraints
Training-time augmentation currently disables the frozen feature cache because image features would otherwise be tied to one changing view. This issue needs precomputed views for data comparison without model training, network access, or GPU use. The outputs are datasets, not report artifacts, and stay ignored by git. The operation writes substantial generated image data when run on the full real training folders.
## Alternatives considered
- Apply augmentation during training — retains epoch-level variation but prevents frozen feature cache reuse.
- Use one fixed torchvision composition — not selected because the required behavior chooses one operation uniformly for each image and logs that choice.
- Transform held-out images along with training — rejected because validation and test are evaluation inputs and must remain unchanged.
## Trade-offs
The disk views consume storage and each adjusted image is JPEG-compressed once. Fixed examples allow cache reuse, but they do not provide a newly sampled view each epoch. Mixed mode doubles the training file count while adjusted-only mode preserves it.
## Key parameters
- Seed `0` (`scripts/make_augmented_view.py::main`) by default; changing it changes operation choices and sampled values.
- Crop scale `[0.8, 1.0]` (`src/ie4483/augmented_view.py::_adjust`): a smaller lower bound allows tighter crops and more subject loss.
- Rotation range `[-15, 15]` degrees (`src/ie4483/augmented_view.py::_adjust`): a wider range creates stronger orientation changes.
- Brightness, contrast, saturation factors `[0.8, 1.2]` (`src/ie4483/augmented_view.py::_adjust`): wider ranges create stronger colour changes.
- Crop output size `224x224` (`src/ie4483/augmented_view.py::OUTPUT_SIZE`): changing it affects model input consistency.
- JPEG quality `95` (`src/ie4483/augmented_view.py::make_augmented_view`): lower quality reduces disk use and image fidelity.
## Assumptions
The training folders contain supported image files directly in each class folder, and the labels are represented by `cat` and `dog`. The data root also contains `val` and `test` directories in the documented layout. The current real images use distinct stems so `_aug.jpg` does not collide with another source filename.
## Edge cases
- Handled: missing input or class folder raises a clear error; unsupported mode and output nested inside input are rejected; absent validation or test trees are skipped.
- Not handled: recursive class-folder inputs, non-image files in training folders, and collisions between an original filename and a generated `_aug.jpg` name are not normalized or repaired.
## Not verified
The generated real validation/test file hashes were not independently compared after materialization. Real data processing completed and folder counts were checked; unit tests verify byte-identical held-out copies on generated fixtures. No training or model scores were run.
## Changes outside the scope
- `.gitignore`: explicitly required by the issue's Run instruction so `data_aug/` and `data_mixed/` remain untracked.
## How it was verified
`tests/test_augmented_view.py` passed both tests. The full CPU suite passed: 81 tests in 136.40 seconds, with one existing learning-rate scheduler warning. Ruff passed on the three changed Python files. Both requested commands completed on `data/` with seed 0 and reported 19,995 adjusted images: 4,988 flips, 5,133 rotations, 4,997 crops, and 4,877 colour jitters. The resulting per-class counts were checked: `data_aug` has 9,998 cat and 9,997 dog files; `data_mixed` has 19,996 cat and 19,994 dog files.
## To change this
To preserve more context in crops, raise the lower crop scale from `0.8` in `src/ie4483/augmented_view.py::_adjust`; the cost is weaker crop variation. To change color strength, narrow or widen the factor interval there. To change the output crop dimensions, update `src/ie4483/augmented_view.py::OUTPUT_SIZE` and ensure the model configuration expects that size.
## Rollback
Remove `src/ie4483/augmented_view.py`, `scripts/make_augmented_view.py`, and `tests/test_augmented_view.py`, then remove the README usage paragraph and the two `.gitignore` entries. The generated `data_aug/` and `data_mixed/` folders are ignored local data and can be removed independently; no training config or model depends on them.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-27 01:36 UTC for issue #24 attempt 1; compared HEAD..working tree (base 04317a3). Records are claims to check against the code, not proof.

**Change summary**

- `(root)/`: 1 file(s), 0 component(s), +2/-0 lines
- `scripts/`: 1 file(s), 0 component(s), +29/-0 lines
- `src/ie4483/`: 1 file(s), 0 component(s), +82/-0 lines
- plus 1 test file(s), 1 doc file(s)
- new files: `scripts/make_augmented_view.py`, `src/ie4483/augmented_view.py`

**Removed symbols**

None.

**Scope**

Declared: `scripts/make_augmented_view.py`, `src/ie4483/augmented_view.py`, `tests/test_augmented_view.py`, `README.md`, `docs/**`. Verdict: violation.
- out of scope: `.gitignore` (modified); explained in the record: yes

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

3 checked by the tool, 1 tested in this run, 17 claimed only (of 21 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 3 checked, 0 tested, 3 claimed
- Key parameters: 0 checked, 0 tested, 6 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/augmented_view.py::make_augmented_view` — dataset writer (named code exists and changed)
  - [tested] How it was verified: `tests/test_augmented_view.py` passed both tests. The full CPU suite passed: 81 tests i... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/augmented_view.py::_adjust` — one adjustment (named code exists and changed)
  - [checked] Components: `scripts/make_augmented_view.py::main` — CLI (named code exists and changed)
  - [claimed] Components: `tests/test_augmented_view.py` — deterministic and layout coverage (names no code)
  - [claimed] Key parameters: Seed `0` (`scripts/make_augmented_view.py::main`) by default; changing it changes opera... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing input or class folder raises a clear error; unsupported mode and outpu... (names no code)
  - [claimed] Components: `README.md — usage (names no code)
  - [claimed] Key parameters: Crop scale `[0.8, 1.0]` (`src/ie4483/augmented_view.py::_adjust`): a smaller lower boun... (no cited constant with a matching value)
  - [claimed] Components: `.gitignore` — generated data folders (names no code)
  - full per-claim list (21) in the JSON record

**Consistency**

0 error(s), 1 warning(s) between the rationale above and the change.
- WARN: 'Main design choice' names no alternative; say which 1-2 real options were possible and why this one won

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0024-augmented-data-branch.md` gives the commit.
<!-- verified-facts:end -->
