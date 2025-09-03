# EHR-R1: Reinforcement Learning for Electronic Health Record SQL Generation

EHR-R1 is a reinforcement learning framework for training SQL generation models on Electronic Health Record (EHR) data, specifically designed for the MIMIC-IV dataset using GRPO.

## 🎯 Overview

EHR-R1 addresses the unique challenges of medical data querying through reinforcement learning. The framework uses GRPO to fine-tune language models, enabling them to understand complex healthcare terminology, temporal relationships, and multi-table joins typical in EHR databases.

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

### Configuration

Copy the sample environment file and configure your settings:

```bash
# Copy sample environment configuration
cp .env.sample .env

# Edit configuration (optional)
vim .env
```

### Training a Model

```bash
# Basic GRPO training
uv run train

# Training with custom parameters
uv run train \
  --data_path data/mimic_iv/train/data.json \
  --model_name MPX0222forHF/SQL-R1-3B \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --output_dir ./outputs \
  --use_wandb

# Training with Weights & Biases logging
uv run train \
  --use_wandb \
  --wandb_project ehr-r1-training \
  --wandb_run_name my_experiment
```

### Model Evaluation

```bash
# Basic evaluation
uv run evaluate

# Evaluate specific model
uv run evaluate --model_name MPX0222forHF/SQL-R1-3B

# Custom dataset and parameters
uv run evaluate \
  --dataset_path data/mimic_iv/test/data.json \
  --num_samples 100 \
  --db_path data/mimic_iv/mimic_iv.sqlite

# Multiple samples with execution accuracy
uv run evaluate --num_samples 500 --db_path data/mimic_iv/mimic_iv.sqlite
```

## 📊 Results Organization

Results are organized by model and timestamp:

```
results/
└── SQL-R1-3B/
    └── run_20250903_143000/
        ├── results.json
        └── evaluation.log
```

## 📁 Project Structure

```
src/ehr_r1/
├── training/
│   └── grpo_trainer.py      # GRPO trainer implementation
├── models/
│   └── reward_model.py      # Reward model for RL training
├── evaluation/
│   └── evaluator.py         # Model evaluation utilities
├── utils/
│   ├── evaluate.py          # Main evaluation script
│   ├── schema.py            # MIMIC-IV schema
│   ├── sql_executor.py      # SQL execution
│   └── config.py            # Configuration management
├── templates/
│   └── *.jinja2             # Prompt templates
├── train.py                 # Main training script
data/mimic_iv/
├── test/
│   ├── data.json            # Questions
│   └── label.json           # Target queries
└── mimic_iv.sqlite          # Database
```

## 📋 Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7.0+ (for GRPO training)
- vLLM 0.10.1+ (for fast inference)

See `pyproject.toml` for complete dependency list.

## 🛠️ Development

### Data Preprocessing
You can follow guidelines by [EHRSQL-2024](https://github.com/glee4810/ehrsql-2024) to obtain the dataset.
Process MIMIC-IV data for training:

```bash
# Navigate to preprocessing directory
cd preprocess

# Run preprocessing script
uv run python preprocess_db.py \
  --data_dir /path/to/mimic-iv/csv \
  --db_name mimic_iv \
  --out_dir ../data \
  --num_patient 1000
```

### Code Quality

```bash
# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.