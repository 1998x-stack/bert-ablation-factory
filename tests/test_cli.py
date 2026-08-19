import subprocess
import sys
import yaml
from pathlib import Path

from tests.conftest import PROJECT_ROOT


def _run_cli(mod: str, cfg_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", mod, "--cfg", str(cfg_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=180,
    )


def _write_cfg(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _pretrain_cfg(out, model_dir, objective):
    return {
        "SEED": 1, "OUTPUT_DIR": str(out), "FP16": False,
        "LOG_EVERY": 1, "SAVE_EVERY": 1, "EVAL_EVERY": 100000,
        "GRAD_ACCUM_STEPS": 1,
        "MODEL": {"name": model_dir},
        "DATA": {"source": "synthetic", "n": 8, "max_seq_len": 16},
        "ABLATION": {"objective": objective,
                     "mask_strategy": "80_10_10", "use_bilstm_head": False},
        "OPTIM": {"lr": 1e-4, "weight_decay": 0.0, "betas": [0.9, 0.999],
                  "eps": 1e-8, "warmup_steps": 0},
        "TRAIN": {"per_device_batch_size": 2, "max_steps": 2,
                  "eval_steps": 100000, "save_steps": 1},
    }


def test_pretrain_mlm_nsp(tmp_path, tiny_model):
    out = tmp_path / "runs_mlm_nsp"
    p = _write_cfg(tmp_path / "mlm_nsp.yaml",
                   _pretrain_cfg(out, tiny_model["pretrain"], "mlm_nsp"))
    r = _run_cli("bert_ablation_factory.cli.pretrain", p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert any((out / "pretrain").iterdir())


def test_pretrain_mlm_only(tmp_path, tiny_model):
    out = tmp_path / "runs_mlm_only"
    p = _write_cfg(tmp_path / "mlm_only.yaml",
                   _pretrain_cfg(out, tiny_model["masked"], "mlm_only"))
    r = _run_cli("bert_ablation_factory.cli.pretrain", p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert any((out / "pretrain").iterdir())


def test_pretrain_ltr(tmp_path, tiny_model):
    out = tmp_path / "runs_ltr"
    p = _write_cfg(tmp_path / "ltr.yaml",
                   _pretrain_cfg(out, tiny_model["lmhead"], "ltr"))
    r = _run_cli("bert_ablation_factory.cli.pretrain", p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert any((out / "pretrain").iterdir())


def test_finetune_classification(tmp_path, tiny_model):
    out = tmp_path / "runs_cls"
    cfg = {
        "SEED": 1, "OUTPUT_DIR": str(out), "FP16": False, "LOG_EVERY": 1,
        "MODEL": {"name": tiny_model["cls"]},
        "TASK": {"name": "glue_sst2"},
        "DATA": {"source": "synthetic", "n": 6, "max_seq_len": 16},
        "OPTIM": {"lr": 5e-5, "weight_decay": 0.0, "betas": [0.9, 0.999],
                  "eps": 1e-8, "warmup_steps": 0},
        "TRAIN": {"per_device_batch_size": 2, "num_epochs": 1,
                  "eval_steps": 1, "save_steps": 1, "RESTARTS": 1, "RESUME": False},
    }
    p = _write_cfg(tmp_path / "cls.yaml", cfg)
    r = _run_cli("bert_ablation_factory.cli.finetune_classification", p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert list((out / "sst2").iterdir()), "classification run dir should exist"


def test_finetune_qa(tmp_path, tiny_model):
    out = tmp_path / "runs_qa"
    cfg = {
        "SEED": 1, "OUTPUT_DIR": str(out), "FP16": False, "LOG_EVERY": 1,
        "MODEL": {"name": tiny_model["bert"]},
        "TASK": {"name": "squad"},
        "DATA": {"source": "synthetic", "n": 4, "max_seq_len": 64,
                 "doc_stride": 16, "max_query_len": 16},
        "ABLATION": {"use_bilstm_head": False},
        "OPTIM": {"lr": 5e-5, "weight_decay": 0.0, "betas": [0.9, 0.999],
                  "eps": 1e-8, "warmup_steps": 0},
        "TRAIN": {"per_device_batch_size": 2, "num_epochs": 1},
    }
    p = _write_cfg(tmp_path / "qa.yaml", cfg)
    r = _run_cli("bert_ablation_factory.cli.finetune_qa", p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "squad_v1" / "bert_squad_v1.pt").exists()


def test_checkpoint_roundtrip(tmp_path):
    import torch
    from torch.optim import AdamW
    from bert_ablation_factory.modeling.heads import ClassificationHead
    from bert_ablation_factory.trainer.checkpoint import (
        save_checkpoint, load_checkpoint,
    )

    m1 = ClassificationHead(32, 3)
    m2 = ClassificationHead(32, 3)
    opt1 = AdamW(m1.parameters(), lr=1e-3)
    opt2 = AdamW(m2.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LinearLR(opt1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    save_checkpoint(tmp_path / "ck.pt", m1, opt1, sched, scaler, epoch=2, step=11)
    epoch, step = load_checkpoint(tmp_path / "ck.pt", m2, opt2, sched, scaler)
    assert (epoch, step) == (2, 11)
    # model weights match after round-trip
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(),
                                  m2.state_dict().items()):
        assert torch.equal(v1, v2)