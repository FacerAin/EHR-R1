"""Main training script."""

import argparse
from typing import Optional

from ehr_r1.data.ehrsql_dataset import EHRSQLDataset
from ehr_r1.training.grpo_trainer import EHRSQLGRPOTrainer
from ehr_r1.utils.config import TrainingConfig, DataConfig, ModelConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EHR-R1 model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Base model name",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Initialize configurations
    training_config = TrainingConfig(
        model_name=args.model_name,
        num_train_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )
    
    data_config = DataConfig(
        train_data_path=args.data_path,
    )
    
    model_config = ModelConfig(
        model_name=args.model_name,
    )
    
    # Load dataset
    dataset = EHRSQLDataset(
        data_path=data_config.train_data_path,
        max_length=data_config.max_length,
    )
    
    # Initialize trainer
    trainer = EHRSQLGRPOTrainer(
        model_name=training_config.model_name,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        use_wandb=training_config.use_wandb,
    )
    
    # Setup models and trainer
    trainer.setup_models()
    trainer.setup_trainer(dataset)
    
    # Start training
    trainer.train(
        dataset=dataset,
        num_epochs=training_config.num_train_epochs,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        output_dir=training_config.output_dir,
    )


if __name__ == "__main__":
    main()