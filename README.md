# IE4483 Artificial Intelligence and Data Mining Project

Dogs vs Cats image classification by transfer learning from ResNet-18/50 with a
custom classifier head, followed by CIFAR-10 and an imbalanced CIFAR-10 study.
This repository is building the training and evaluation pipeline; the initial
setup does not train on real data or download weights.

## Environment

Use the shared environment at `C:/Users/AgenticDev/ml-venv/Scripts/python.exe`.
The data root is `data/` by default, can be set with `IE4483_DATA`, or can be
configured in `configs/base.yaml`. Data, run outputs, and model weights are
ignored by git.

## Tests

Run the CPU test suite with:

```powershell
C:/Users/AgenticDev/ml-venv/Scripts/python.exe -m pytest -q
```
