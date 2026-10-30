from pathlib import Path

import pytest

from ie4483.config import Config, config_to_dict, dump_config, load_config

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


def test_config_round_trip(tmp_path):
    config = load_config(BASE)
    output = dump_config(config, tmp_path / "run")
    restored = load_config(output)
    assert isinstance(restored, Config)
    assert config_to_dict(restored) == config_to_dict(config)
    assert restored.data.classes == ["cat", "dog"]


def test_config_overrides_are_typed():
    config = load_config(BASE, ["train.seed=17", "model.pretrained=false", "data.classes=[cat, dog]"])
    assert config.train.seed == 17
    assert config.model.pretrained is False
    assert config.data.classes == ["cat", "dog"]


def test_unknown_config_key_is_rejected(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("train:\n  se ed: 12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration key 'train.se ed'"):
        load_config(config_file)


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown configuration key 'train.mystery'"):
        load_config(BASE, ["train.mystery=3"])


def test_data_root_uses_config_then_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("IE4483_DATA", str(tmp_path / "environment"))
    config = load_config(BASE)
    assert config.data_root == tmp_path / "environment"

    configured = tmp_path / "configured"
    config_file = tmp_path / "configured.yaml"
    config_file.write_text(f"data:\n  root: '{configured}'\n", encoding="utf-8")
    assert load_config(config_file).data_root == configured


def test_model_freeze_mode_accepted_and_validated(tmp_path) -> None:
    import pytest as _pytest

    from ie4483.config import load_config
    from ie4483.model import build_model

    path = tmp_path / "c.yaml"
    path.write_text("model:\n  pretrained: false\n  freeze_mode: last_block\n", encoding="utf-8")
    config = load_config(path)
    assert config.model.freeze_mode == "last_block"
    assert build_model(config).freeze_mode == "last_block"
    assert load_config(path, ["model.freeze_mode=null"]).model.freeze_mode is None
    with _pytest.raises(ValueError, match="freeze_mode"):
        load_config(path, ["model.freeze_mode=partial"])
