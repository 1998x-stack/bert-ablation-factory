import torch
from bert_ablation_factory.modeling.heads import ClassificationHead, SpanHead
from bert_ablation_factory.modeling.bilstm import BiLSTMEncoder


def test_classification_head_shapes():
    out = ClassificationHead(32, 3)(torch.randn(2, 10, 32), torch.ones(2, 10))
    assert out.shape == (2, 3)


def test_span_head_shapes():
    s, e = SpanHead(32)(torch.randn(2, 10, 32), torch.ones(2, 10))
    assert s.shape == (2, 10) and e.shape == (2, 10)


def test_bilstm_keeps_shape():
    assert BiLSTMEncoder(32)(torch.randn(2, 10, 32)).shape == (2, 10, 32)


def test_bilstm_classification_head():
    out = ClassificationHead(32, 3, use_bilstm=True)(torch.randn(2, 10, 32), torch.ones(2, 10))
    assert out.shape == (2, 3)