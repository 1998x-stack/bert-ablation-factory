import torch
from bert_ablation_factory.data.collators import (
    MLMConfig, MLMOnlyCollator, LTRCollator,
)


def test_mlm_masks_valid_not_pad():
    torch.manual_seed(0)
    c = MLMOnlyCollator(MLMConfig(mask_strategy="100_mask", pad_token_id=0, mask_token_id=4))
    n = 50
    ids = list(range(1, 1 + n))
    batch = c([{"input_ids": ids, "token_type_ids": [0] * n,
                "attention_mask": [1] * n}])
    inp = batch["input_ids"][0]
    lab = batch["labels"][0]
    assert (inp == 4).sum() > 0                 # some tokens masked
    masked = lab != -100                        # masked positions carry real labels
    assert masked.sum() > 0                     # some tokens have real labels
    assert (inp[masked] == 4).all()             # every masked position is a MASK token (100_mask)


def test_ltr_collator_shifts_and_pads_last():
    c = LTRCollator()
    b = c([{"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4}])
    labels = b["labels"][0]
    assert labels[0] == 2 and labels[1] == 3 and labels[2] == 4
    assert labels[-1] == -100
