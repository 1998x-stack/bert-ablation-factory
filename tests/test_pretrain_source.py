def test_pretrain_synthetic_stream(tiny_tokenizer):
    from bert_ablation_factory.cli.pretrain import build_books_wiki_stream
    cfg = {"DATA": {"source": "synthetic", "n": 4}}
    stream = build_books_wiki_stream(cfg, tiny_tokenizer, max_len=16)
    first = next(stream)
    assert first["input_ids"] and first["attention_mask"]
    assert "next_sentence_label" in first