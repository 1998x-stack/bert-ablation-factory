from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import evaluate
import numpy as np

from ..registry import TASKS


# GLUE 主指标映射（选择单一主指标用于 early-stop/选最优 run）
# 参考 GLUE 官方：CoLA->matthews_correlation；STS-B->pearson/spearman，常以 pearson 为主
MAIN_METRIC = {
    "sst2": "accuracy",
    "mnli": "accuracy",              # 我们使用 matched 的 accuracy 作为主指标
    "qnli": "accuracy",
    "qqp": "f1",                     # 同时也会回报 accuracy
    "mrpc": "f1",                    # 同时也会回报 accuracy
    "rte": "accuracy",
    "cola": "matthews_correlation",
    "stsb": "pearson",
}


def _sentence_keys(task: str) -> Tuple[str, Optional[str]]:
    """返回不同 GLUE 任务的句子列名（单句或句对）。"""
    if task in ("sst2", "cola"):
        return "sentence", None
    if task in ("qnli", "rte"):
        return "question", "sentence"
    if task in ("mrpc", "qqp"):
        return "sentence1", "sentence2"
    if task in ("mnli",):
        return "premise", "hypothesis"
    if task in ("stsb",):
        return "sentence1", "sentence2"
    raise ValueError(f"Unsupported GLUE task: {task}")


def _num_labels_and_type(task: str) -> Tuple[int, str]:
    """返回 (num_labels, problem_type)。STS-B 是回归，其余是分类。"""
    if task == "stsb":
        return 1, "regression"
    # 其余分类任务的 label 数
    num = {
        "sst2": 2, "cola": 2, "mrpc": 2, "qqp": 2, "qnli": 2, "rte": 2,
        "mnli": 3,  # entailment/neutral/contradiction
    }[task]
    return num, "single_label_classification"


def _metric_for(task: str):
    """加载对应的 GLUE 度量器（evaluate）。"""
    return evaluate.load("glue", task)


def _preprocess_builder(tokenizer: PreTrainedTokenizerBase, task: str, max_len: int):
    s1, s2 = _sentence_keys(task)

    def fn(batch):
        if s2 is None:
            enc = tokenizer(batch[s1], truncation=True, max_length=max_len, padding="max_length")
        else:
            enc = tokenizer(batch[s1], batch[s2], truncation=True, max_length=max_len, padding="max_length")
        # label 直接保留
        enc["labels"] = batch["label"]
        return enc

    return fn


def _split_names(task: str) -> Tuple[str, Optional[str]]:
    """返回 dev split 名称（MNLI 有 matched/mismatched）。"""
    if task == "mnli":
        return "validation_matched", "validation_mismatched"
    return "validation", None


def _postprocess_logits(task: str, logits) -> np.ndarray:
    """将模型输出转为 metric 需要的预测值（分类取 argmax，回归直接取值）。"""
    if task == "stsb":
        return logits.squeeze(-1)
    return logits.argmax(axis=-1)


@TASKS.register("glue_sst2")
@TASKS.register("glue_mnli")
@TASKS.register("glue_qnli")
@TASKS.register("glue_qqp")
@TASKS.register("glue_mrpc")
@TASKS.register("glue_rte")
@TASKS.register("glue_cola")
@TASKS.register("glue_stsb")
def build_glue_task(cfg: Dict[str, Any], tokenizer: PreTrainedTokenizerBase):
    """统一构建 GLUE 任务的数据与评测。

    返回:
        {
          "task_name": str,                    # 纯 task 名（不含 glue_ 前缀）
          "train_ds": Dataset,
          "dev_ds": Dataset or dict[str, Dataset],  # MNLI 情况下有两个 dev
          "num_labels": int,
          "problem_type": "regression" | "single_label_classification",
          "metric": evaluate.Metric,
          "main_metric": str,                  # 主指标 key
        }
    """
    full = cfg["TASK"]["name"]          # e.g., "glue_sst2"
    assert full.startswith("glue_"), "TASK.name 必须以 glue_ 开头"
    task = full.split("_", 1)[1]

    raw = load_dataset("glue", task)
    max_len = int(cfg["DATA"]["max_seq_len"])
    fn = _preprocess_builder(tokenizer, task, max_len)

    train = raw["train"].map(fn, batched=True, remove_columns=raw["train"].column_names)
    train.set_format(type="torch")

    dev_name, dev_name2 = _split_names(task)
    dev = raw[dev_name].map(fn, batched=True, remove_columns=raw[dev_name].column_names)
    dev.set_format(type="torch")

    dev_bundle = dev
    if dev_name2 is not None:
        dev2 = raw[dev_name2].map(fn, batched=True, remove_columns=raw[dev_name2].column_names)
        dev2.set_format(type="torch")
        dev_bundle = {"matched": dev, "mismatched": dev2}

    num_labels, problem_type = _num_labels_and_type(task)
    metric = _metric_for(task)
    main_metric = MAIN_METRIC[task]

    return {
        "task_name": task,
        "train_ds": train,
        "dev_ds": dev_bundle,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "metric": metric,
        "main_metric": main_metric,
    }


def compute_glue_metrics(task: str, metric, logits, labels) -> Dict[str, float]:
    """对单一 split 计算 GLUE 指标（分类回归兼容）。"""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = _postprocess_logits(task, logits)
    res = metric.compute(predictions=preds, references=labels)
    return {k: float(v) for k, v in res.items()}


def pick_main_score(task: str, metric_result: Dict[str, float]) -> float:
    """抽取主指标分数，用于 early-stop 或多重启选最优。"""
    key = MAIN_METRIC[task]
    return float(metric_result.get(key, -1e9))
