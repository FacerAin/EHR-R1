# EHR-R1: Reinforcement Learning for Electronic Health Record SQL Generation

A reinforcement learning framework for training SQL generation models on Electronic Health Record (EHR) data using the TRL (Transformer Reinforcement Learning) library and GRPO (Group Relative Policy Optimization).

## 🎯 Overview

EHR-R1 focuses on training language models to generate accurate SQL queries for electronic health record systems. The project implements state-of-the-art reinforcement learning techniques to improve model performance on complex medical data queries.

## 🚀 Quick Start

### Installation

This project uses `uv` for dependency management:

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

**Alternative one-line setup:**
```bash
# Create environment and install in one command
uv sync --python 3.11

# For development with all tools
uv sync --python 3.11 --extra dev
```

### Training a Model

```bash
# Basic training
uv run train --data_path /path/to/ehrsql/data

# Training with custom configuration
uv run train \
  --data_path /path/to/data \
  --model_name MPX0222forHF/SQL-R1-3B \
  --num_epochs 5 \
  --output_dir ./my_outputs
```

### Model Evaluation

The evaluation script uses vLLM for fast inference and supports local dataset files.

```bash
# Evaluate SQL-R1-3B model with default dataset (data/mimic_iv/test/data.json)
uv run python src/ehr_r1/evaluate.py

# Evaluate with custom model
uv run python src/ehr_r1/evaluate.py --model_name seeklhy/OmniSQL-7B

# Evaluate with multiple GPUs for faster inference
uv run python src/ehr_r1/evaluate.py --tensor_parallel_size 2 --num_samples 500

# Use custom dataset file
uv run python src/ehr_r1/evaluate.py --dataset_path /path/to/your/data.json

# Adjust generation parameters
uv run python src/ehr_r1/evaluate.py --temperature 0.1 --max_length 2048 --num_samples 100

# Use different prompt templates
uv run python src/ehr_r1/evaluate.py --prompt_template simple_sql_prompt.jinja2
uv run python src/ehr_r1/evaluate.py --prompt_template few_shot_prompt.jinja2
```

**Evaluation Features:**
- **vLLM Integration**: Fast batch inference with GPU parallelism
- **Local Dataset Support**: Works with JSON files containing question-SQL pairs
- **Official OmniSQL Format**: Follows the exact prompt format from official OmniSQL repository
- **Template-based Prompts**: Jinja2 template system for easy prompt customization and management
- **Detailed Database Schema**: Uses complete CREATE TABLE statements with column descriptions and examples
- **Comprehensive Metrics**: Exact match accuracy and SQL component analysis
- **Structured Logging**: Detailed progress tracking and debugging information
- **Flexible Configuration**: Adjustable sampling parameters and model settings

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
│   ├── prompts.py              # Prompt templates and rendering
│   ├── sql_executor.py         # SQL execution utilities
│   ├── logger.py               # Logging configuration
│   ├── schema.py               # Database schema definitions
│   └── __init__.py
├── templates/                   # Jinja2 template files
│   ├── omnisql_prompt.jinja2   # Official OmniSQL format
│   ├── simple_sql_prompt.jinja2 # Simple SQL prompt
│   └── few_shot_prompt.jinja2  # Few-shot learning prompt
├── train.py                     # Main training script
├── evaluate.py                  # Main evaluation script
├── inference.py                 # Model inference
├── evaluate_execution.py        # SQL execution evaluation
└── __init__.py
preprocess/                      # Data preprocessing utilities
├── preprocess_utils.py         # General preprocessing functions
├── preprocess_db.py            # Database preprocessing
├── preprocess_db_mimic_iv.py   # MIMIC-IV specific preprocessing
└── preprocess.sh               # Preprocessing script
data/                           # Dataset files
└── mimic_iv/                   # MIMIC-IV dataset
    └── test/
        └── data.json           # Test dataset
```

## ✨ Key Features

- **🔄 Reinforcement Learning**: GRPO (Group Relative Policy Optimization) training using TRL
- **🏥 EHR Specialization**: Designed specifically for Electronic Health Record SQL generation
- **🎯 Reward-Based Training**: Custom reward model for evaluating SQL query quality
- **📊 Comprehensive Evaluation**: Multiple metrics including execution accuracy and exact match
- **📈 Experiment Tracking**: Weights & Biases integration for monitoring training progress
- **🔧 Flexible Configuration**: Easy-to-use configuration system for experiments
- **⚡ High Performance**: Optimized for efficient training and inference using vLLM
- **📝 Comprehensive Logging**: Structured logging throughout the application

## 🛠️ Development

### Data Preprocessing

Process MIMIC-IV data for training and evaluation:

```bash
# Process MIMIC-IV database
cd preprocess
python preprocess_db.py \
  --data_dir /path/to/mimic-iv/csv \
  --db_name mimic_iv \
  --out_dir ../data \
  --num_patient 1000 \
  --deid \
  --timeshift \
  --start_year 2100 \
  --time_span 3 \
  --current_time "2100-12-31 23:59:59"
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

### Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/ehr_r1
```

## 📋 Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7.0+
- vLLM 0.10.1+ (for fast inference)

See `pyproject.toml` for complete dependency list.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.