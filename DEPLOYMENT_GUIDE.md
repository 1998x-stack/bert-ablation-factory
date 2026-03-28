# Production Deployment Guide

## Overview

This guide covers deploying the BERT Ablation Factory to production environments, including cloud infrastructure, monitoring, and scaling.

## Prerequisites

### Hardware Requirements

**Minimum (Development):**
- CPU: 8 cores
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: 1x NVIDIA T4 (16GB) or equivalent

**Recommended (Production):**
- CPU: 32+ cores
- RAM: 256+ GB
- Storage: 1 TB+ NVMe SSD
- GPU: 8x NVIDIA A100 (40GB) or equivalent
- Network: 10 Gbps

### Software Requirements

- Python: 3.8-3.11
- CUDA: 11.8+ (if using GPU)
- PyTorch: 2.1+
- Git: 2.30+
- Docker: 24.0+ (optional)

## Installation

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/bert-ablation-factory.git
cd bert-ablation-factory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import bert_ablation_factory; print('✅ Installation successful')"
```

### 2. GPU Setup (if applicable)

```bash
# Verify CUDA installation
nvidia-smi

# Test PyTorch CUDA
torch.cuda.is_available()

# Set CUDA visible devices (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3. Data Preparation

```bash
# Create data directory
mkdir -p /mnt/storage/bert-data

# Set permissions
chmod 755 /mnt/storage/bert-data

# Pre-download datasets (optional but recommended)
python -c "
from datasets import load_dataset
load_dataset('bookcorpusopen', split='train')
load_dataset('wikipedia', '20220301.en', split='train')
"
```

## Configuration

### 1. Production Configuration

Copy the production configuration:

```bash
cp production-config.yaml configs/production.yaml

# Edit as needed
nano configs/production.yaml
```

Key settings to adjust:
- `OUTPUT_DIR`: Set to your storage path
- `TRAIN.per_device_batch_size`: Adjust for GPU memory
- `OPTIM.lr`: Learning rate for your task
- `LOGGING.wandb`: Configure Weights & Biases

### 2. Environment Variables

Create `.env` file:

```bash
# Required
export OUTPUT_DIR=/mnt/storage/bert-ablation-runs
export WANDB_API_KEY=your_wandb_key
export HF_TOKEN=your_huggingface_token

# Optional
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
```

Load environment:
```bash
source .env
```

## Running Training

### 1. Pretraining

```bash
# Single GPU
python -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml \
    --objective mlm_nsp

# Multi-GPU (if configured)
torchrun --nproc_per_node=4 \
    -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml \
    --objective mlm_nsp
```

### 2. Finetuning

```bash
# GLUE task
python -m bert_ablation_factory.cli.finetune_classification \
    --cfg configs/production.yaml \
    --task glue_sst2

# SQuAD
python -m bert_ablation_factory.cli.finetune_qa \
    --cfg configs/production.yaml \
    --task squad_v1
```

### 3. Monitoring Training

```bash
# TensorBoard
tensorboard --logdir $OUTPUT_DIR/tensorboard

# Watch logs
tail -f $OUTPUT_DIR/logs/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Monitoring and Alerting

### 1. Weights & Biases Integration

The production config includes W&B integration. Monitor runs at:
https://wandb.ai/your-entity/bert-ablation-factory

### 2. Custom Metrics

Track these key metrics:
- Training loss
- Validation loss/accuracy
- Learning rate
- GPU memory usage
- Training throughput (samples/sec)
- Checkpoint save/load times

### 3. Alerting Rules

Set up alerts for:
- Training loss increasing
- GPU memory > 90%
- Disk space < 100GB
- Training stalled (>1 hour no progress)
- NaN/Inf in metrics

## Checkpoint Management

### 1. Automatic Checkpointing

Checkpoints are saved automatically every `save_steps`. Configure:

```yaml
CHECKPOINT:
  save_total_limit: 5  # Keep only last 5
  load_best_model_at_end: true
```

### 2. Manual Checkpointing

```bash
# Save checkpoint manually
curl -X POST http://localhost:5000/save_checkpoint

# List checkpoints
ls -lt $OUTPUT_DIR/checkpoints/

# Resume from checkpoint
python -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml \
    --resume_from_checkpoint $OUTPUT_DIR/checkpoints/latest
```

### 3. Checkpoint Validation

```bash
# Verify checkpoint integrity
python scripts/verify_checkpoint.py \
    --checkpoint $OUTPUT_DIR/checkpoints/checkpoint-10000
```

## Scaling and Distributed Training

### 1. Multi-GPU Training

```bash
# Using torch.distributed.launch
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml
```

### 2. Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml

# Node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m bert_ablation_factory.cli.pretrain \
    --cfg configs/production.yaml
```

### 3. DeepSpeed Integration

For very large models, enable DeepSpeed:

```yaml
DISTRIBUTED:
  enabled: true
  deepspeed: "configs/ds_config.json"
```

Create `configs/ds_config.json`:
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

## Performance Optimization

### 1. Mixed Precision Training

Already enabled in production config:
```yaml
MODEL:
  fp16: true
```

### 2. Gradient Accumulation

For effective large batch sizes:
```yaml
TRAIN:
  per_device_batch_size: 16
  gradient_accumulation_steps: 4  # Effective batch = 64
```

### 3. Data Loading Optimization

```yaml
DATA:
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true
```

### 4. Memory Optimization

For large models:
```yaml
MEMORY:
  gradient_checkpointing: true  # Trade compute for memory
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
# Enable gradient checkpointing
# Use gradient accumulation
```

**2. Slow Training**
```bash
# Increase num_workers
# Enable mixed precision
# Check GPU utilization
# Profile with: python -m cProfile -o profile.stats train.py
```

**3. Checkpoint Resume Fails**
```bash
# Verify checkpoint files exist
# Check disk space
# Validate checkpoint: python scripts/verify_checkpoint.py
```

**4. Reproducibility Issues**
```bash
# Set deterministic operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; torch.use_deterministic_algorithms(True)"
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m bert_ablation_factory.cli.pretrain --cfg configs/production.yaml
```

## Security Considerations

### 1. Data Security

- Use encrypted storage for sensitive data
- Implement access controls for data directories
- Regular backups of important checkpoints

### 2. Model Security

- Secure checkpoint storage
- Access control for model artifacts
- Version control for production models

### 3. Network Security

- Use VPN for distributed training
- Firewall rules for training ports
- Secure API endpoints for monitoring

## Maintenance

### 1. Regular Tasks

- **Daily:** Monitor training progress, check GPU health
- **Weekly:** Review metrics, clean old checkpoints
- **Monthly:** Update dependencies, backup important models
- **Quarterly:** Performance review, infrastructure upgrades

### 2. Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update carefully
pip install --upgrade torch transformers datasets

# Test after updates
pytest tests/
```

### 3. Disk Space Management

```bash
# Clean old checkpoints (keep last 5)
find $OUTPUT_DIR/checkpoints -name "checkpoint-*" -type d \
  | sort -r | tail -n +6 | xargs rm -rf

# Clean old logs
find $OUTPUT_DIR/logs -name "*.log" -mtime +30 -delete
```

## Support and Resources

### Documentation
- Full API docs: `docs/api.md`
- Configuration guide: `docs/configuration.md`
- Troubleshooting: `docs/troubleshooting.md`

### Getting Help
1. Check troubleshooting guide
2. Review logs in `$OUTPUT_DIR/logs/`
3. Run tests to verify installation
4. Check GitHub issues
5. Contact support team

### Reporting Issues
Include:
- Full error message and stack trace
- Configuration file (sanitized)
- System information (GPU, RAM, OS)
- Training logs
- Steps to reproduce

---

**Next Steps:**
1. Follow installation steps
2. Configure production settings
3. Run test training job
4. Set up monitoring
5. Begin production training

**For questions or issues:**
- GitHub Issues: https://github.com/your-org/bert-ablation-factory/issues
- Documentation: https://bert-ablation-factory.readthedocs.io/
