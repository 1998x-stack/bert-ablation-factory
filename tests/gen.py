"""Shared offline builders for the test suite: tiny vocab, tokenizer, and BERT config."""
from pathlib import Path

from transformers import BertConfig, BertTokenizer

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB_SIZE = 1000


def make_config(**overrides) -> BertConfig:
    cfg = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        mask_token_id=4,
    )
    cfg.update(overrides)
    return BertConfig(**cfg)


def make_tokenizer(base: Path) -> BertTokenizer:
    """Write a tiny vocab + tokenizer into ``base`` and return the tokenizer.

    The vocab has 1000 entries: [PAD],[UNK],[CLS],[SEP],[MASK], then w5..w999
    (pad=0, unk=1, cls=2, sep=3, mask=4). IDs stay in [0,999] == model vocab_size.
    """
    base.mkdir(parents=True, exist_ok=True)
    vocab = SPECIALS + [f"w{i}" for i in range(5, VOCAB_SIZE)]
    (base / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizer(vocab_file=str(base / "vocab.txt"))
    tok.save_pretrained(str(base))
    return tok