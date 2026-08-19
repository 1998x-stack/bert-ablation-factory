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


def test_glue_unknown_source_raises(tiny_tokenizer):
    from bert_ablation_factory.tasks.glue import build_glue_task
    import pytest
    with pytest.raises(ValueError):
        build_glue_task(_cfg("bogus"), tiny_tokenizer)