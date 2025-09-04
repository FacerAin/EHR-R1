# EHR-R1: Reinforcement Learning for EHR SQL Generation

Reinforcement learning framework for EHR SQL generation using GRPO.

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
cp .env.sample .env
```

## Usage

```bash
# Train a model with GRPO
uv run train --use_wandb --num_epochs 3

# Evaluate model performance  
uv run evaluate --model_name MPX0222forHF/SQL-R1-3B --num_samples 1000
```

## Multi-GPU Training

### Setup Accelerate

1. Configure accelerate (first time only):
```bash
accelerate config
```

2. For multi-GPU training on a single machine, select:
   - Compute environment: This machine
   - Distributed type: multi-GPU
   - Number of processes: number of GPUs you want to use
   - Mixed precision: bf16

### Launch Multi-GPU Training

Use accelerate launch instead of python:
```bash
accelerate launch --config_file accelerate_config.yaml uv run train \
    --model-name MPX0222forHF/SQL-R1-3B \
    --data-path data/ehrsql/train \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8
```

Or with automatic GPU detection:
```bash
accelerate launch --num_processes 4 uv run train \
    --model-name MPX0222forHF/SQL-R1-3B \
    --data-path data/ehrsql/train \
    --epochs 3
```

## 📊 Evaluation Results

### Model Performance on MIMIC-IV (934 samples)

| Model | Size | Execution Accuracy | HCAcc@90% |
|-------|------|-------------------| ------ |
| **EHR-R1-7B (Ours)** | 7B | TBA | TBA | 
| **EHR-R1-3B (Ours)** | 3B | TBA | TBA |
| **Arctic-Text2SQL-R1-7B** | 7B | 36.1% | TBA |
| **OmniSQL-7B** | 7B | 34.0% | TBA |
| **SQL-R1-7B** | 7B | **37.3%** | TBA |
| **SQL-R1-3B** | 3B | 24.0% | TBA |


## Data

Obtain MIMIC-IV dataset following [EHRSQL-2024](https://github.com/glee4810/ehrsql-2024) guidelines.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7.0+
- vLLM 0.10.1+

## License

MIT
