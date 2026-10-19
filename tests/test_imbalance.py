import torch
from torch.utils.data import Dataset

from ie4483.evaluate import _scores
from ie4483.imbalance import (
    FocalLoss,
    balanced_sampler,
    class_weights,
    imbalanced_subset,
)


class Labels(Dataset):
    def __init__(self):
        self.labels = torch.tensor([0] * 20 + [1] * 10 + [2] * 10).numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(index), int(self.labels[index])


def test_seeded_imbalanced_subset_counts_and_weights():
    data = Labels()
    subset, counts = imbalanced_subset(data, {1: 0.2}, seed=7)
    assert counts == {0: 20, 1: 2, 2: 10}
    assert subset.indices == imbalanced_subset(data, {1: 0.2}, seed=7)[0].indices
    weights = class_weights(data.labels[subset.indices], 3)
    assert weights[1] > weights[2] > weights[0]
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_weighted_sampler_is_balanced_and_focal_gamma_zero_is_ce():
    labels = torch.tensor([0] * 90 + [1] * 10)
    sampler = balanced_sampler(labels, seed=11)
    sampled = labels[list(sampler)]
    assert abs(int((sampled == 0).sum()) - int((sampled == 1).sum())) < 30
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    torch.testing.assert_close(FocalLoss(0)(logits, target), torch.nn.functional.cross_entropy(logits, target))


def test_macro_metrics_average_classes_not_examples():
    metrics = _scores([0, 0, 0, 1], [0, 0, 0, 0], 0.0, ["majority", "minority"])
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_recall"] == 0.5
    assert metrics["macro_f1"] == 3 / 7
