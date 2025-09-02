# EHR-R1: Reinforcement Learning for Electronic Health Record SQL Generation

A reinforcement learning framework for training SQL generation models on Electronic Health Record (EHR) data using the TRL (Transformer Reinforcement Learning) library and GRPO (Group Relative Policy Optimization).

## 🎯 Overview

EHR-R1 focuses on training language models to generate accurate SQL queries for electronic health record systems. The project implements state-of-the-art reinforcement learning techniques to improve model performance on complex medical data queries.

## 🚀 Quick Start

### Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

### Training a Model

```bash
# Basic training
train --data_path /path/to/ehrsql/data

# Training with custom configuration
train \
  --data_path /path/to/data \
  --model_name microsoft/DialoGPT-medium \
  --num_epochs 5 \
  --output_dir ./my_outputs
```

### Model Evaluation

```bash
# Evaluate trained model
evaluate --model_path ./outputs/checkpoint-1000 --test_data_path /path/to/test/data
```

## 📁 Project Structure

```
src/ehr_r1/
├── data/                        # Data processing and loading
│   ├── ehrsql_dataset.py       # EHRSQL dataset implementation
│   └── __init__.py
├── models/                      # Model definitions
│   ├── reward_model.py         # Reward model for RL training
│   └── __init__.py
├── training/                    # Training components
│   ├── grpo_trainer.py         # GRPO trainer implementation
│   └── __init__.py
├── evaluation/                  # Evaluation and metrics
│   ├── evaluator.py            # Model evaluation utilities
│   └── __init__.py
├── utils/                       # Utility modules
│   ├── config.py               # Configuration management
│   ├── prompts.py              # Prompt templates
│   ├── sql_executor.py         # SQL execution utilities
│   └── __init__.py
├── train.py                     # Main training script
├── evaluate.py                  # Main evaluation script
├── inference.py                 # Model inference
├── evaluate_execution.py        # SQL execution evaluation
└── __init__.py
preprocess/                      # Data preprocessing utilities
├── preprocess_utils.py         # General preprocessing functions
├── preprocess_db.py            # Database preprocessing
└── preprocess_db_mimic_iv.py   # MIMIC-IV specific preprocessing
```

## ✨ Key Features

- **🔄 Reinforcement Learning**: GRPO (Group Relative Policy Optimization) training using TRL
- **🏥 EHR Specialization**: Designed specifically for Electronic Health Record SQL generation
- **🎯 Reward-Based Training**: Custom reward model for evaluating SQL query quality
- **📊 Comprehensive Evaluation**: Multiple metrics including execution accuracy and exact match
- **📈 Experiment Tracking**: Weights & Biases integration for monitoring training progress
- **🔧 Flexible Configuration**: Easy-to-use configuration system for experiments
- **⚡ High Performance**: Optimized for efficient training and inference

## 🛠️ Development

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

### Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/ehr_r1
```

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7.0+

See `pyproject.toml` for complete dependency list.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.