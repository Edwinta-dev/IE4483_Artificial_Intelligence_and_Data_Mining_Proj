import csv
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ie4483 import submission
from scripts import predict as predict_script


def test_prediction_csvs_sort_ids_and_match_submission_format(tmp_path):
    submission.write_prediction_csvs(
        tmp_path, [10, 2], [[0.2, 0.8], [0.9, 0.1]], ["cat", "dog"], 2
    )
    assert (tmp_path / "submission.csv").read_bytes() == b"id,label\n2,0\n10,1\n"
    with (tmp_path / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["id", "p_cat", "p_dog"], ["2", "0.9", "0.1"], ["10", "0.2", "0.8"]
        ]


@pytest.mark.parametrize(
    ("ids", "probs", "classes", "count", "message"),
    [
        ([1], [[0.5, 0.5]], ["cat", "dog"], 2, "row count"),
        ([1, 1], [[0.5, 0.5]] * 2, ["cat", "dog"], 2, "unique"),
        ([1], [[0.5, 0.5]], ["dog", "cat"], 1, "class order"),
    ],
)
def test_prediction_validation_errors(tmp_path, ids, probs, classes, count, message):
    with pytest.raises(ValueError, match=message):
        submission.write_prediction_csvs(tmp_path, ids, probs, classes, count)


@pytest.mark.parametrize("tta", [None, "flip"])
def test_predict_reads_generated_test_images_and_writes_sorted_rows(tmp_path, monkeypatch, tta):
    (tmp_path / "test").mkdir()
    for image_id, color in ((10, (220, 0, 0)), (2, (0, 220, 0))):
        Image.new("RGB", (8, 8), color).save(tmp_path / "test" / f"{image_id}.png")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(classes=["cat", "dog"]), data_root=tmp_path,
        train=SimpleNamespace(seed=7), eval=SimpleNamespace(batch_size=2),
    )

    class TinyModel(torch.nn.Module):
        def forward(self, images):
            mean = images.mean(dim=(1, 2, 3))
            return torch.stack((1 - mean, mean), dim=1)

    monkeypatch.setattr(predict_script, "load_training_config", lambda _: (config, {"num_workers": 0}))
    monkeypatch.setattr(predict_script, "build_eval_transform", lambda _: lambda image: torch.from_numpy(
        __import__("numpy").array(image, copy=True)
    ).permute(2, 0, 1).float().div_(255))
    monkeypatch.setattr(predict_script, "build_model", lambda _: TinyModel())
    monkeypatch.setattr(predict_script.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(predict_script, "seed_everything", lambda _: None)
    predict_script.predict(str(run), tta=tta)
    assert (run / "submission.csv").read_bytes().startswith(b"id,label\n2,")
    with (run / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        assert [row["id"] for row in csv.DictReader(stream)] == ["2", "10"]
