def test_squad_synthetic(tiny_tokenizer):
    from bert_ablation_factory.data.squad import load_squad_v1
    cfg = {"DATA": {"source": "synthetic", "max_seq_len": 64,
                    "doc_stride": 16, "max_query_len": 16, "n": 4}}
    train, dev = load_squad_v1(tiny_tokenizer, cfg)
    assert len(train) >= 2 and len(dev) >= 2
    b = train[0]
    for k in ["input_ids", "attention_mask", "start_positions", "end_positions"]:
        assert k in b

    # Offline context overflows into >1 window (long wNNN context).
    assert len(train) > 1

    cls_id = tiny_tokenizer.cls_token_id  # == 2
    first_sp = int(b["start_positions"])
    first_ep = int(b["end_positions"])

    # REAL, non-CLS answer alignment in the first crafted window: the answer is
    # the single in-vocab token "w12" (id 12) sitting early in the context.
    assert first_sp != cls_id, "first window start must not be CLS"
    assert first_ep != cls_id, "first window end must not be CLS"
    # Single-token span aligned to the in-vocab answer token id.
    assert first_sp == first_ep, "expected a single-token answer span"
    assert int(b["input_ids"][first_sp]) == 12, b["input_ids"][first_sp]