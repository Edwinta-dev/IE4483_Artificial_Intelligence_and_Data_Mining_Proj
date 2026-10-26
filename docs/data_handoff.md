# Real data handoff audit

The team handoff in `raw-data/datasets/` is materialized under the ignored `data/` root by `src/ie4483/preprocess_images.py`. The converter reads every supported source image, applies EXIF orientation, converts to RGB, and writes 224×224 JPEGs using Pillow `ImageOps.fit` with LANCZOS. This is a centre-fit crop: it fills the square and crops excess width or height. It processes splits in train, validation, test order and keeps only the first output with each converted JPEG's SHA-256 digest. This prevents exact duplicates from crossing the training/evaluation boundary. A rerun clears old supported image files from the generated destination before writing, preventing stale outputs from surviving.

The project's runtime evaluation transform instead resizes the shorter edge and takes a centre crop (`src/ie4483/transforms.py::build_eval_transform`). Training can add configured random resized crop and horizontal flip before normalization (`src/ie4483/transforms.py::build_train_transform`). The handoff crop is permanently applied before these pipeline transforms, and JPEG encoding is also irreversible.

Counts below are from `data/audit.json` produced by `scripts/audit_data.py --root data --compare raw-data`:

| Split | Class | Count |
|---|---|---:|
| train | cat | 9,998 |
| train | dog | 9,997 |
| val | cat | 2,499 |
| val | dog | 2,500 |
| test | unlabelled | 500 |

The converter read 25,500 source files with zero failures. Six duplicate outputs were omitted. The audit reports zero unreadable files, zero exact duplicates within training, zero train/validation duplicates, and zero train/test duplicates. All remaining images decode and have 224×224 dimensions. The raw-vs-output comparison lists differ because the comparison root is `raw-data` while the source images are nested under `raw-data/datasets`; corresponding converted paths therefore have different prefixes and extensions. It is a relative-path listing, not a conversion-failure report.

Run from the repository root:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe src/ie4483/preprocess_images.py
C:/Users/AgenticDev/ml-venv/Scripts/python.exe scripts/audit_data.py --root data --compare raw-data
```
