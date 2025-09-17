# BERT Ablation Factory (Pluggable)

- Pretrain (MLM+NSP / MLM-only / LTR) and Finetune (SST-2, SQuAD v1.1).
- Toggle ablations via YAML: NSP on/off, 80/10/10 masking, BiLSTM heads, LTR (optional).

## Quickstart

```bash
pip install -r requirements.txt

# Pretrain (MLM+NSP)
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/mlm_nsp_base.yaml

# Finetune SST-2 (GLUE)
python -m bert_ablation_factory.cli.finetune_classification --cfg configs/finetune/glue_sst2_base.yaml

# Finetune SQuAD v1.1
python -m bert_ablation_factory.cli.finetune_qa --cfg configs/finetune/squad_v1_base.yaml
```
TensorBoard: `tensorboard --logdir runs`