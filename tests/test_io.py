import yaml
import pytest
from bert_ablation_factory.utils.io import load_yaml, merge_dict


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    assert load_yaml(p) == {"a": 1}


def test_load_yaml_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_merge_dict():
    assert merge_dict({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}