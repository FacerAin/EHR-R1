"""Configuration utilities."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Training configuration."""

    model_name: str = "microsoft/DialoGPT-medium"
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    grpo_epochs: int = 1
    max_prompt_length: int = 256
    beta: float = 0.1
    max_length: int = 512
    num_train_epochs: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./outputs"
    use_wandb: bool = True
    wandb_project: str = "ehr-r1"


@dataclass
class DataConfig:
    """Data configuration."""

    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    max_length: int = 512
    preprocessing_num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "microsoft/DialoGPT-medium"
    reward_model_name: Optional[str] = None
    use_peft: bool = True
    peft_config: Optional[Dict[str, Any]] = None


class ConfigManager:
    """Configuration manager."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
