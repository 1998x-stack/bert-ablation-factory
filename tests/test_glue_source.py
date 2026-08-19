import json


def _cfg(source, **extra):
    c = {
        "TASK": {"name": "glue_sst2"},
        "MODEL": {"name": "x"},
        "DATA": {"max_seq_len": 8, "source": source},
    }
    c["DATA"].update(extra)
    return c


def test_glue_synthetic_returns_bundle(tiny_tokenizer):
    from bert_ablation_factory.tasks.glue import build_glue_task
    bundle = build_glue_task(_cfg("synthetic"), tiny_tokenizer)
    assert bundle["task_name"] == "sst2"
    assert bundle["num_labels"] == 2
    assert bundle["problem_type"] == "single_label_classification"
    assert len(bundle["train_ds"]) >= 2
    assert len(bundle["dev_ds"]) >= 1
    assert bundle["main_metric"] == "accuracy"


def test_glue_json(tmp_path, tiny_tokenizer):
    from bert_ablation_factory.tasks.glue import build_glue_task
    lines = [{"sentence": f"tiny json sentence {i}", "label": i % 2}
             for i in range(5)]
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    dev_path.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    cfg = _cfg("json", train_path=str(train_path), dev_path=str(dev_path))
    bundle = build_glue_task(cfg, tiny_tokenizer)
    assert len(bundle["train_ds"]) >= 1
    assert len(bundle["dev_ds"]) >= 1
    assert bundle["task_name"] == "sst2"


def test_glue_json_missing_path_raises(tiny_tokenizer):
    from bert_ablation_factory.tasks.glue import build_glue_task
    import pytest
    with pytest.raises(ValueError):
        build_glue_task(_cfg("json", train_path="train.jsonl"), tiny_tokenizer)


def test_glue_unknown_source_raises(tiny_tokenizer):
    from bert_ablation_factory.tasks.glue import build_glue_task
    import pytest
    with pytest.raises(ValueError):
        build_glue_task(_cfg("bogus"), tiny_tokenizer)
