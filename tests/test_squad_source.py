def test_squad_synthetic(tiny_tokenizer):
    from bert_ablation_factory.data.squad import load_squad_v1
    cfg = {"DATA": {"source": "synthetic", "max_seq_len": 16,
                    "doc_stride": 8, "max_query_len": 8, "n": 4}}
    train, dev = load_squad_v1(tiny_tokenizer, cfg)
    assert len(train) >= 2 and len(dev) >= 2
    b = train[0]
    for k in ["input_ids", "attention_mask", "start_positions", "end_positions"]:
        assert k in b