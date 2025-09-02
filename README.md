# EHR-R1: Reinforcement Learning for Electronic Health Record SQL Generation

EHR-R1 is a reinforcement learning framework for training SQL generation models on Electronic Health Record (EHR) data, specifically designed for the MIMIC-IV dataset using GRPO (Group Relative Policy Optimization).

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

### Training a Model

```bash
# Basic training with GRPO
uv run train --data_path /path/to/ehrsql/data

# Training with custom configuration
uv run train \
  --data_path /path/to/data \
  --model_name MPX0222forHF/SQL-R1-3B \
  --num_epochs 5 \
  --output_dir ./my_outputs
```

### Model Evaluation

```bash
# Basic evaluation
python -m src.ehr_r1.utils.evaluate

# Evaluate specific model
python -m src.ehr_r1.utils.evaluate --model_name MPX0222forHF/SQL-R1-3B

# Custom dataset
python -m src.ehr_r1.utils.evaluate --dataset_path data/mimic_iv/test/data.json

# Multiple samples with execution accuracy
python -m src.ehr_r1.utils.evaluate --num_samples 500 --db_path data/mimic_iv/mimic_iv.sqlite
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

## ✨ Key Features

- **🔄 Reinforcement Learning**: GRPO (Group Relative Policy Optimization) training using TRL
- **🏥 EHR Specialization**: Designed for Electronic Health Record SQL generation
- **🎯 Reward-Based Training**: Custom reward model for evaluating SQL query quality
- **📊 Comprehensive Evaluation**: Exact match and execution accuracy metrics
- **📈 Experiment Tracking**: Weights & Biases integration for monitoring training progress
- **🗃️ MIMIC-IV Integration**: Native support for MIMIC-IV database structure

## 🛠️ Reinforcement Learning Framework

### GRPO Training
EHR-R1 uses Group Relative Policy Optimization (GRPO) to train models with reward-based feedback:

- **Policy Model**: Base language model fine-tuned for SQL generation
- **Reward Model**: Evaluates SQL query quality and correctness
- **Group Optimization**: Efficient batch-wise policy updates

### Training Pipeline
1. **Supervised Fine-tuning**: Initial training on question-SQL pairs
2. **Reward Model Training**: Train reward model to score SQL quality
3. **GRPO Optimization**: Reinforce correct SQL generation patterns
4. **Evaluation**: Test on held-out EHR datasets

## 🔍 Evaluation Metrics

### Exact Match Accuracy
Compares normalized SQL strings after parsing.

### Execution Accuracy
Executes queries on the actual database and compares results:
- **Execution Accuracy**: Percentage of correct results
- **Predicted Success Rate**: Percentage of queries that execute successfully
- **Ground Truth Success Rate**: Reference query success rate

### Sample Output
```
=== EVALUATION RESULTS ===
Exact Match Accuracy: 0.0000
Execution Accuracy: 0.2500
Predicted Success Rate: 0.8000
Ground Truth Success Rate: 1.0000
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

Process MIMIC-IV data for training:

```bash
cd preprocess
python preprocess_db.py \
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