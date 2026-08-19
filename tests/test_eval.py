import torch
from bert_ablation_factory.trainer.eval import accuracy, squad_em_f1


def test_accuracy():
    pred = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 1])
    assert accuracy(pred, target) == 1.0


def test_accuracy_empty():
    assert accuracy(torch.zeros(0, 2), torch.zeros(0)) == 1.0


def test_squad_em_f1_exact():
    r = squad_em_f1(torch.tensor([2, 2]), torch.tensor([4, 4]),
                    torch.tensor([2, 2]), torch.tensor([4, 4]))
    assert r["em"] == 1.0 and r["f1"] == 1.0