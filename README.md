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

## 📊 Evaluation Results

### Model Performance on MIMIC-IV (934 samples)

| Model | Size | Execution Accuracy |
|-------|------|-------------------|
| **EHR-R1-7B (Ours)** | 7B | TBA |
| **EHR-R1-3B (Ours)** | 3B | TBA |
| **Arctic-Text2SQL-R1-7B** | 7B | 36.1% |
| **OmniSQL-7B** | 7B | 34.0% |
| **SQL-R1-7B** | 7B | **37.3%** |
| **SQL-R1-3B** | 3B | 24.0% |


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