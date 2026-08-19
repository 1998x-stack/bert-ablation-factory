import pytest


@pytest.fixture(scope="module")
def tiny_tok_dir(tmp_path_factory):
    from tests.gen import make_tokenizer
    base = tmp_path_factory.mktemp("tok")
    make_tokenizer(base)
    return base


def test_build_tokenizer_returns_bert_tokenizer(tiny_tok_dir):
    from bert_ablation_factory.data.tokenization import build_tokenizer
    tok = build_tokenizer({"MODEL": {"name": tiny_tok_dir}})
    assert tok.mask_token_id == 4
    assert tok.pad_token_id == 0
    assert tok.cls_token_id == 2
    assert tok.vocab_size == 1000


def test_build_tokenizer_missing_name_raises():
    from bert_ablation_factory.data.tokenization import build_tokenizer
    with pytest.raises(ValueError):
        build_tokenizer({"MODEL": {}})


def test_build_tokenizer_bad_name_raises(tmp_path):
    from bert_ablation_factory.data.tokenization import build_tokenizer
    with pytest.raises(OSError):
        build_tokenizer({"MODEL": {"name": str(tmp_path / "does-not-exist")}})