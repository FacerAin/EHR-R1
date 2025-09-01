# EHR-R1: Reinforcement Learning for EHRSQL using TRL

EHR-R1 is a project for training reinforcement learning models for SQL generation on the EHRSQL dataset using TRL (Transformer Reinforcement Learning).

## Installation

This project uses the `uv` package manager:

```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

## Usage

### Training

```bash
# Basic training
ehr-r1-train --data_path /path/to/ehrsql/data

# Training with configuration options
ehr-r1-train --data_path /path/to/data --model_name microsoft/DialoGPT-medium --num_epochs 5 --output_dir ./my_outputs
```

### Evaluation

```bash
# Model evaluation
ehr-r1-eval --model_path ./outputs/checkpoint-1000 --test_data_path /path/to/test/data
```

## Project Structure

```
src/ehr_r1/
├── data/                    # Data loading and preprocessing
│   ├── ehrsql_dataset.py   # EHRSQL dataset class
│   └── __init__.py
├── models/                  # Model definitions
│   ├── reward_model.py     # Reward model
│   └── __init__.py
├── training/               # Training modules
│   ├── grpo_trainer.py    # GRPO trainer
│   └── __init__.py
├── evaluation/             # Evaluation modules
│   ├── evaluator.py       # Evaluation metrics
│   └── __init__.py
├── utils/                  # Utility modules
│   ├── config.py          # Configuration management
│   └── __init__.py
├── train.py               # Training entry point
├── evaluate.py            # Evaluation entry point
└── __init__.py
```

## Key Features

- **TRL-based GRPO Training**: GRPO (Group Relative Policy Optimization) training using the TRL library
- **EHRSQL Dataset Support**: Processing Electronic Health Record SQL datasets
- **Reward Model**: Reward model for evaluating SQL query quality
- **Comprehensive Evaluation**: Various evaluation metrics including execution accuracy and exact match
- **Weights & Biases Integration**: Training monitoring and logging

## Development

```bash
# Code formatting
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```