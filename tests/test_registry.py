import pytest
from bert_ablation_factory.registry import Registry


def test_registry_roundtrip():
    r = Registry("t")
    @r.register("a")
    def a():
        return 1
    assert r.get("a")() == 1
    assert r.keys() == ["a"]


def test_registry_duplicate_raises():
    r = Registry("t2")
    @r.register("a")
    def a():
        return 1
    with pytest.raises(KeyError):
        r.register("a")(lambda: 2)


def test_registry_missing_raises():
    with pytest.raises(KeyError):
        Registry("t3").get("nope")