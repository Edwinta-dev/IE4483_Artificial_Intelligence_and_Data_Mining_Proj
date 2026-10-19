import os
import random

import numpy as np
import torch

from ie4483.seed import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_values():
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic


def test_seed_everything_configures_determinism(monkeypatch):
    seed_everything(3, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(3, deterministic=False)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    assert not torch.are_deterministic_algorithms_enabled()
