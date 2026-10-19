# 0003: Dataset loading and seeded subsets
<!-- verify-banner:begin -->
> ⚠ Agent-written record: its claims are hypotheses to verify against the code, not ground truth. Evidence levels in Verified facts: [checked] confirmed by the tool · [tested] cited test exists and passed in this run · [claimed] not verified.
<!-- verify-banner:end -->
Status: implemented · Issue: #3 · Date: 2026-10-19
## In short
The package can now read labelled class-folder splits and an unlabelled numeric-ID test folder. It can reproducibly limit each class, skip and count unreadable files, write a manifest of selected labelled files, and batch samples with CUDA-aware memory pinning.
## How it works
1. Give the loader a data root, split name, and ordered class list. It looks for image files directly inside each class folder and uses each class's position in that list as its integer label.
2. Before samples are exposed, each candidate image is checked for readability. Unreadable images are logged, counted, and excluded. If a per-class limit is requested, a seeded random choice selects the files independently for each class; leaving the limit unset keeps all readable files.
3. The labelled dataset optionally writes a CSV listing the split, class, and full path for every selected file. This makes the actual subset inspectable in a run folder.
4. The test dataset reads image files from the test folder, uses each numeric filename stem as its ID, and sorts by the numeric value. It does not create labels.
5. Samples open as RGB and become channel-first float tensors scaled to zero through one by default. A caller may provide its own transform. The loader helper batches a dataset, defaults to no worker processes, and pins memory only when CUDA is available.
## Main design choice
The datasets use Pillow and PyTorch directly, with a small NumPy conversion for the default tensor transform. A torchvision ImageFolder-based design was a plausible alternative, but it expects labelled directory trees and does not cover the unlabelled numeric-ID contract or manifest/subset tracking as directly. A second alternative was to return PIL images and require every caller to provide transforms; the built-in conversion won because default PyTorch batching cannot collate PIL images. Callers needing normalization or resizing can supply a transform.
## Traps: why the obvious fix is wrong
- Sorting test names as strings would put `10` before `2`; IDs are parsed and sorted as integers.
- Sampling one global subset can starve a class; the limit is applied separately to each configured class.
- A corrupt file can fail later during batching and abort an epoch; readability is checked while building the index and excluded then.
- Labels must follow the caller's class order; alphabetically sorting classes would silently change dog from label 1 when class order differs.
## Components
- `src/ie4483/data.py::ImageFolderDataset` — labelled split, subset, manifest
- `src/ie4483/data.py::UnlabelledTestDataset` — numeric IDs
- `src/ie4483/data.py::make_dataloader` — batching and pinning
- `src/ie4483/data.py::_default_transform` — RGB tensor conversion
- `tests/test_data.py` — generated data behavior
- `configs/base.yaml` — existing classes, seed, workers
## Breaks or needs follow-up
None. No existing data-loading API existed.
## Problem and constraints
Issue #3 adds the data input layer for Dogs vs Cats and later class-folder datasets. Tests must use tiny generated data and CPU only; there must be no implicit network, weights, real data, or GPU dependency. Existing config already holds an ordered class list, training seed, and zero default evaluation workers, so no config key was needed.
## Alternatives considered
- torchvision folder datasets — less custom code, but does not provide the required test-ID dataset and selected-file manifest contract.
- Return PIL images by default — preserves raw decoding but breaks default DataLoader collation without a caller transform.
## Trade-offs
Index construction decodes every candidate once for validation, adding startup I/O. This avoids a later corrupt-image crash; it may matter for much larger datasets, where a cached integrity scan could reduce repeated startup cost.
## Key parameters
- `n_per_class=None` (`src/ie4483/data.py::ImageFolderDataset`) — keep all readable files; setting an integer caps each class, and lowering it reduces sample count.
- `seed=42` (`src/ie4483/data.py::ImageFolderDataset`) — deterministic subset default; changing it selects another subset.
- `num_workers=0` (`src/ie4483/data.py::make_dataloader`) — conservative Windows default; increasing it may improve throughput at added process overhead.
- `IMAGE_SUFFIXES` (`src/ie4483/data.py::IMAGE_SUFFIXES`) — recognized raster extensions; adding a suffix expands discovered files.
## Assumptions
Class directory names match the configured class names. Test image stems are numeric; nonnumeric files are skipped with a warning. Readability at indexing is assumed to remain stable until retrieval.
## Edge cases
- Handled: missing split/class/test folders, corrupt images, non-image files, numeric ID ordering, custom class order, and empty subsets.
- Not handled: files changing or disappearing after indexing; such a race can still raise during item retrieval.
## Not verified
No CUDA device was required or tested, and no real dataset was loaded. Pinning behavior is selected from the runtime CUDA availability flag but was exercised only in this CPU test run.
## Changes outside the scope
None. `README.md` and `configs/base.yaml` needed no changes because this issue adds no command or config key and existing settings already cover the required values.
## How it was verified
`tests/test_data.py` checks configured labels, deterministic per-class subsets, corrupt-file exclusion and count, manifest rows, numeric test ID ordering, and loader batching. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q` passed 10 tests. `C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m ruff check src/ie4483/data.py tests/test_data.py` is also run for the changed Python files.
## To change this
To use a different pixel normalization, replace `src/ie4483/data.py::_default_transform` or supply a transform. To recognize another raster extension, add it to `src/ie4483/data.py::IMAGE_SUFFIXES`. To change subset selection, alter the seed passed to `src/ie4483/data.py::ImageFolderDataset`.
## Rollback
Remove `src/ie4483/data.py`, `tests/test_data.py`, and this decision record. No existing module imports the new API.

<!-- verified-facts:begin (generated by integrity record; do not edit) -->
## Verified facts

Generated 2026-10-19 02:08 UTC for issue #3 attempt 3; compared HEAD..working tree (base 8370c7e). Records are claims to check against the code, not proof.

**Change summary**

- `src/ie4483/`: 1 file(s), 0 component(s), +154/-0 lines
- plus 1 test file(s)
- new files: `src/ie4483/data.py`

**Removed symbols**

None.

**Scope**

Declared: `src/ie4483/data.py`, `configs/base.yaml`. Verdict: pass.

**Supervisor checks passed**

whitespace/conflict-marker check, project test suite (CPU), python compileall (changed dirs), scope check, dangling reference check

**Evidence**

4 checked by the tool, 1 tested in this run, 14 claimed only (of 19 claims). [checked] = the named code exists and changed, or the cited value matches; [tested] = the cited test exists and the test gate passed in this run; [claimed] = not verified.
- Components: 4 checked, 0 tested, 2 claimed
- Key parameters: 0 checked, 0 tested, 4 claimed
- Edge cases: 0 checked, 0 tested, 1 claimed
- How it was verified: 0 checked, 1 tested, 0 claimed
- Traps: 0 checked, 0 tested, 4 claimed
- Assumptions: 0 checked, 0 tested, 1 claimed
- Trade-offs: 0 checked, 0 tested, 1 claimed
- Not verified: 0 checked, 0 tested, 1 claimed
  - [checked] Components: `src/ie4483/data.py::ImageFolderDataset` — labelled split, subset, manifest (named code exists and changed)
  - [tested] How it was verified: `tests/test_data.py` checks configured labels, deterministic per-class subsets, corrupt... (test gate passed: project test suite (CPU))
  - [checked] Components: `src/ie4483/data.py::UnlabelledTestDataset` — numeric IDs (named code exists and changed)
  - [checked] Components: `src/ie4483/data.py::make_dataloader` — batching and pinning (named code exists and changed)
  - [checked] Components: `src/ie4483/data.py::_default_transform` — RGB tensor conversion (named code exists and changed)
  - [claimed] Components: `tests/test_data.py` — generated data behavior (names no code)
  - [claimed] Key parameters: `n_per_class=None` (`src/ie4483/data.py::ImageFolderDataset`) — keep all readable files... (no cited constant with a matching value)
  - [claimed] Edge cases: Handled: missing split/class/test folders, corrupt images, non-image files, numeric ID... (names no code)
  - [claimed] Components: `configs/base.yaml` — existing classes, seed, workers (`configs/base.yaml` did not change)
  - [claimed] Key parameters: `seed=42` (`src/ie4483/data.py::ImageFolderDataset`) — deterministic subset default; ch... (no cited constant with a matching value)
  - full per-claim list (19) in the JSON record

**Consistency**

0 error(s), 0 warning(s) between the rationale above and the change.

**Commit**

This record is committed with the change it describes; `git log -1 -- docs/decisions/0003-dataset-loading.md` gives the commit.
<!-- verified-facts:end -->
