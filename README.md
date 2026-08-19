# BERT Ablation Factory (Pluggable)

A flexible BERT ablation study framework that enables configurable pretraining and fine-tuning experiments.
Supports various BERT variants and ablation studies including MLM+NSP, MLM-only, LTR, BiLSTM heads, and different masking strategies.

## Table of Contents
- [Introduction](#introduction)
- [What is BERT Ablation?](#what-is-bert-ablation)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Understanding Configuration Options](#understanding-configuration-options)
- [Project Structure](#project-structure)
- [Learning Resources](#learning-resources)
- [Examples](#examples)
- [Monitoring Training](#monitoring-training)

## Introduction

The BERT Ablation Factory is designed for researchers and practitioners who want to understand how different components of BERT contribute to its performance. Through systematic removal or modification of components, you can analyze their individual impact on the model's capabilities.

This framework is particularly useful for:
- Understanding BERT's internal mechanisms
- Comparing different pretraining objectives
- Testing architectural modifications
- Conducting rigorous ablation studies

## What is BERT Ablation?

Ablation studies in the context of BERT involve systematically removing or modifying specific components to understand their contribution:

- **Masked Language Modeling (MLM)**: Understanding the importance of bidirectional context
- **Next Sentence Prediction (NSP)**: Evaluating the role of sentence-level relationships
- **Bidirectional vs Left-to-Right**: Comparing different training objectives
- **Different Masking Strategies**: Testing various approaches to masking tokens
- **Architecture Modifications**: Adding components like BiLSTM layers

## Features

- **Modular Design**: Pluggable components for easy experimentation
- **Multiple Objectives**: Support for MLM+NSP, MLM-only, and Left-to-Right (LTR) training
- **Flexible Configurations**: YAML-based configuration for easy experiment management
- **Ablation Studies**: Built-in support for various BERT modifications and component analysis
- **Task Support**: Pretraining and fine-tuning for classification (GLUE) and QA (SQuAD) tasks
- **BiLSTM Integration**: Option to add BiLSTM layers on top of BERT representations
- **Comprehensive Logging**: Detailed metrics and TensorBoard integration

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of deep learning and NLP concepts
- Familiarity with PyTorch and Transformers library

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd bert-ablation-factory

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

### 1. Pretrain with Different Objectives

```bash
# Pretrain with MLM+NSP (standard BERT objective)
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/mlm_nsp_base.yaml

# Pretrain with MLM-only (no NSP)
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/mlm_no_nsp_base.yaml

# Pretrain with Left-to-Right objective
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/ltr_no_nsp_base.yaml
```

### 2. Fine-tune on Downstream Tasks

```bash
# Fine-tune on SST-2 sentiment classification (GLUE task)
python -m bert_ablation_factory.cli.finetune_classification --cfg configs/finetune/glue_sst2_base.yaml

# Fine-tune on SQuAD v1.1 question answering
python -m bert_ablation_factory.cli.finetune_qa --cfg configs/finetune/squad_v1_base.yaml
```

### 3. Run with Custom Configurations

You can modify the YAML configuration files to experiment with different settings:

```bash
# Custom configuration with BiLSTM head
python -m bert_ablation_factory.cli.finetune_classification --cfg my_custom_config.yaml
```

## Understanding Configuration Options

The framework uses YAML configuration files for maximum flexibility. Here's what you can control:

### Pretraining Configurations:
- **Masking Strategy**: Choose between 80/10/10 (80% [MASK], 10% random, 10% unchanged) or 100% [MASK]
- **Training Objective**: MLM+NSP, MLM-only, or LTR (Left-to-Right)
- **Model Architecture**: Base BERT or with additional BiLSTM layers
- **Training Parameters**: Learning rate, batch size, number of steps, etc.

### Fine-tuning Configurations:
- **Task Selection**: Different GLUE tasks or SQuAD
- **Number of Restarts**: Multiple runs with different random seeds for statistical significance
- **Evaluation Metrics**: Task-specific metrics like F1, EM, accuracy
- **Hyperparameters**: Learning rates, epochs, warmup steps, etc.

### DATA.source (dataset injection)

Each data loader reads `DATA.source`:

- `hf` (default): download the standard dataset from HuggingFace (`glue`, `squad`, or the bookcorpusopen+wikipedia stream for pretraining).
- `json`: load local JSON/JSONL files given by `DATA.train_path` and `DATA.dev_path`.
- `synthetic` (test convenience): generate a tiny in-memory dataset — used by the offline unit/CLI smoke tests.

## Project Structure

```
bert-ablation-factory/
├── bert_ablation_factory/          # Main package
│   ├── cli/                       # Command-line interfaces
│   │   ├── pretrain.py           # Pretraining scripts
│   │   ├── finetune_classification.py  # Classification fine-tuning
│   │   └── finetune_qa.py        # Question answering fine-tuning
│   ├── modeling/                  # Model definitions
│   │   ├── build.py              # Model construction utilities
│   │   ├── heads.py              # Task-specific heads
│   │   ├── objectives.py         # Training objectives
│   │   └── bilstm.py             # BiLSTM components
│   ├── data/                      # Data processing
│   │   ├── tokenization.py       # Tokenizer builders
│   │   ├── collators.py          # Batch collation
│   │   ├── glue.py               # GLUE dataset utilities
│   │   └── squad.py              # SQuAD dataset utilities
│   ├── trainer/                   # Training logic
│   │   ├── engine.py             # Training loop
│   │   ├── eval.py               # Evaluation utilities
│   │   ├── checkpoint.py         # Checkpoint management
│   │   ├── optimizer.py          # Optimizer utilities
│   │   └── schedulers.py         # Learning rate schedulers
│   ├── tasks/                     # Task definitions
│   ├── utils/                     # Utility functions
│   └── registry.py                # Component registries
├── configs/                       # Configuration files
│   ├── base.yaml                 # Base configuration
│   ├── pretrain/                 # Pretraining configs
│   └── finetune/                 # Fine-tuning configs
├── tests/                        # Offline unit + CLI smoke tests (conftest, gen, per-module)
└── README.md                     # This file
```

## Testing

The suite runs fully offline — no model, tokenizer, or dataset downloads — using
tiny synthetic data and tiny random-weight BERT checkpoints.

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Learning Resources

### Key Concepts:
1. **BERT Fundamentals**: Understand the original BERT paper and its pretraining objectives
2. **Ablation Studies**: Learn why systematically removing components is valuable for research
3. **Transformer Architecture**: Familiarize yourself with attention mechanisms
4. **Evaluation Metrics**: Know how to interpret results for different tasks

### Recommended Reading:
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)

### Best Practices for Ablation Studies:
1. **Statistical Significance**: Run multiple experiments with different random seeds
2. **Consistent Baselines**: Ensure fair comparison across conditions
3. **Control Variables**: Change only one component at a time when possible
4. **Detailed Logging**: Track all relevant metrics for comprehensive analysis

## Examples

### Example 1: Comparing Pretraining Objectives
Compare MLM+NSP vs MLM-only on downstream performance:

```bash
# Train with both objectives
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/mlm_nsp_base.yaml

# Train with MLM only
python -m bert_ablation_factory.cli.pretrain --cfg configs/pretrain/mlm_no_nsp_base.yaml

# Fine-tune both models on the same task
python -m bert_ablation_factory.cli.finetune_classification --cfg configs/finetune/glue_sst2_base.yaml
```

### Example 2: Testing Different Masking Strategies
Evaluate the impact of different masking approaches:

```bash
# 80/10/10 masking (BERT standard)
# 100% masking (alternative strategy)
# Compare performance on downstream tasks
```

## Monitoring Training

Monitor training progress and metrics with TensorBoard:

```bash
# Start TensorBoard server
tensorboard --logdir runs

# Access the dashboard at http://localhost:6006
```

The TensorBoard dashboard will show:
- Training and validation loss
- Task-specific metrics (accuracy, F1, EM, etc.)
- Learning rate schedule
- Gradient norms (if logged)

## Contributing

We welcome contributions to improve the framework:
- Bug reports and fixes
- New ablation configurations
- Additional tasks and datasets
- Documentation improvements

For major changes, please open an issue first to discuss what you would like to change.