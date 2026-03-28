# BERT Ablation Factory - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you get the BERT Ablation Factory running quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/bert-ablation-factory.git
cd bert-ablation-factory
```

### Step 2: Verify Installation (No Dependencies Required!)

```bash
# Run verification script (works without installing dependencies)
python scripts/verify_installation.py
```

This will check:
- ✅ Python version compatibility
- ✅ Package structure
- ✅ Core imports
- ✅ Configuration files
- ✅ CLI entry points

**Expected output:**
```
🔍 BERT Ablation Factory - Installation Verification
============================================================
✅ Python 3.14.2 - OK
✅ Directory exists: bert_ablation_factory
✅ File exists: bert_ablation_factory/__init__.py
...
🎉 All checks passed! Installation looks good.
```

### Step 3: Install Dependencies (Optional for Full Functionality)

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install core dependencies only
pip install torch numpy pyyaml loguru tqdm tensorboard
```

### Step 4: Run Examples (Works Without Full Dependencies!)

```bash
# Run component examples (demonstrates each part of the system)
python examples/runnable_components.py
```

This will show:
- ✅ Registry system
- ✅ Configuration inheritance
- ✅ BiLSTM encoder
- ✅ Model heads (classification & QA)
- ✅ Data collators
- ✅ Checkpoint management
- ✅ Training loop structure

### Step 5: Run Integration Tests (Works Without Full Dependencies!)

```bash
# Run integration tests (tests end-to-end workflows)
python examples/integration_test.py
```

This tests:
- ✅ Pretraining workflow
- ✅ Finetuning workflow
- ✅ Device management
- ✅ Configuration validation

## Running Training (Requires Dependencies)

### Pretraining

```bash
# MLM + NSP pretraining
python -m bert_ablation_factory.cli.pretrain \
    --cfg configs/pretrain/mlm_nsp_base.yaml

# MLM-only pretraining
python -m bert_ablation_factory.cli.pretrain \
    --cfg configs/pretrain/mlm_only_base.yaml
```

### Finetuning

```bash
# GLUE classification (SST-2)
python -m bert_ablation_factory.cli.finetune_classification \
    --cfg configs/finetune/glue_sst2_base.yaml

# SQuAD QA
python -m bert_ablation_factory.cli.finetune_qa \
    --cfg configs/finetune/squad_v1_base.yaml
```

## Configuration

### Quick Configuration Examples

**Pretraining (short demo):**
```yaml
# configs/pretrain/mlm_nsp_short.yaml
ABLATION:
  objective: mlm_nsp
  mask_strategy: 80_10_10

TRAIN:
  per_device_batch_size: 4
  max_steps: 100  # Short demo
```

**Finetuning (short demo):**
```yaml
# configs/finetune/glue_sst2_short.yaml
TASK:
  name: glue_sst2

TRAIN:
  per_device_batch_size: 4
  num_epochs: 1  # Short demo
```

## Testing

### Run Unit Tests (Requires pytest)

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_registry.py -v
pytest tests/test_bilstm.py -v
pytest tests/test_cli_config.py -v

# Run with coverage
pytest tests/ -v --cov=bert_ablation_factory
```

### Run Without pytest (Alternative)

```bash
# Run verification script (no dependencies needed)
python scripts/verify_installation.py

# Run examples (no dependencies needed)
python examples/runnable_components.py

# Run integration tests (minimal dependencies)
python examples/integration_test.py
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at: http://localhost:6006
```

### Weights & Biases

```bash
# Set API key
export WANDB_API_KEY=your_api_key

# Training will automatically log to W&B
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'transformers'**
```bash
# Install transformers
pip install transformers
```

**2. CUDA out of memory**
```bash
# Reduce batch size in config
TRAIN:
  per_device_batch_size: 8  # Reduce from 32 to 8
```

**3. Slow training**
```bash
# Enable mixed precision
MODEL:
  fp16: true

# Increase data workers
DATA:
  num_workers: 8
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
python -m bert_ablation_factory.cli.pretrain --cfg config.yaml
```

## Performance Tips

### For Development/Small Scale:

```yaml
# Use short sequence length
DATA:
  max_seq_len: 128  # Instead of 512

# Use small batch size
TRAIN:
  per_device_batch_size: 4

# Fewer steps
TRAIN:
  max_steps: 1000
```

### For Production/Large Scale:

```yaml
# Use production config as base
_base_: production-config.yaml

# Adjust for your hardware
TRAIN:
  per_device_batch_size: 32
  gradient_accumulation_steps: 2

# Enable optimizations
MODEL:
  fp16: true
MEMORY:
  gradient_checkpointing: true
```

## Next Steps

### For Research:
1. Read `DEBUGGING_REPORT.md` for implementation details
2. Review `configs/` for ablation study configurations
3. Run experiments with different objectives (mlm_nsp, mlm_only, ltr)

### For Production:
1. Read `DEPLOYMENT_GUIDE.md` for production setup
2. Use `production-config.yaml` as base
3. Set up monitoring (TensorBoard, W&B)
4. Configure checkpoint management

### For Development:
1. Read `QA_ENHANCEMENT_REPORT.md` for code quality
2. Review test suite in `tests/`
3. Contribute improvements

## Documentation

- **Quick Start:** This file
- **Deployment:** `DEPLOYMENT_GUIDE.md`
- **Debugging:** `DEBUGGING_REPORT.md`
- **QA/Testing:** `QA_ENHANCEMENT_REPORT.md`
- **Production:** `PRODUCTION_READINESS_CHECKLIST.md`

## Support

### Getting Help

1. **Check docs first** - Most questions answered in documentation
2. **Run verification** - `python scripts/verify_installation.py`
3. **Check examples** - `python examples/runnable_components.py`
4. **Review tests** - `tests/` directory has many examples
5. **GitHub Issues** - https://github.com/your-org/bert-ablation-factory/issues

### Reporting Issues

Include:
- Python version: `python --version`
- Error message and stack trace
- Config file used
- Steps to reproduce
- System info (GPU, RAM, OS)

## Summary

✅ **Installation:** 1 command (`python scripts/verify_installation.py`)
✅ **Dependencies:** Optional for basic functionality
✅ **Examples:** Runnable without full dependencies
✅ **Tests:** Comprehensive test suite
✅ **Docs:** Complete documentation
✅ **Production:** Ready for deployment

**Total setup time:** ~5 minutes

---

**Ready to start? Run:**
```bash
python scripts/verify_installation.py
```
