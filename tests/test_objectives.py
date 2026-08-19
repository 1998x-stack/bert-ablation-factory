import torch
from bert_ablation_factory.modeling.objectives import LossCombine


def test_loss_combine_mlm_nsp():
    lc = LossCombine(use_mlm=True, use_nsp=True)
    out = {
        "mlm_logits": torch.randn(2, 8, 10),
        "mlm_labels": torch.randint(0, 10, (2, 8)),
        "nsp_logits": torch.randn(2, 2),
        "next_sentence_label": torch.randint(0, 2, (2,)),
        "hidden_states": torch.randn(2, 8, 10),
    }
    total, parts = lc(out)
    assert "mlm" in parts and "nsp" in parts
    assert total.ndim == 0