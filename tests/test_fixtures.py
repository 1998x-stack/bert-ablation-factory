from transformers import BertForPreTraining


def test_tiny_model_offline(tiny_model):
    pt = BertForPreTraining.from_pretrained(tiny_model["pretrain"])
    assert pt.config.hidden_size == 32


def test_tiny_tokenizer_offline(tiny_tokenizer, tiny_model):
    assert tiny_tokenizer.mask_token_id == 4
    assert tiny_tokenizer.pad_token_id == 0
    assert tiny_tokenizer.cls_token_id == 2
    assert tiny_tokenizer.vocab_size == 1000
    # tokenizer resolves from the same dir as the model (MODEL.name)
    from bert_ablation_factory.data.tokenization import build_tokenizer
    tok = build_tokenizer({"MODEL": {"name": tiny_model["pretrain"]}})
    assert tok.mask_token_id == 4