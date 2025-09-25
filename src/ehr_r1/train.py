"""Main training script."""

import argparse
import os
from typing import Optional

from dotenv import load_dotenv

from ehr_r1.data.ehrsql_dataset import EHRSQLDataset
from ehr_r1.training.grpo_trainer import EHRSQLGRPOTrainer
from ehr_r1.utils.config import DataConfig, ModelConfig, TrainingConfig
from ehr_r1.utils.logger import get_logger

logger = get_logger(__name__)


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
        default="data/mimic_iv/train/data.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="data/mimic_iv/valid/data.json",
        help="Path to evaluation data",
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
        default="MPX0222forHF/SQL-R1-3B",
        help="Base model name",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per device train batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/mimic_iv/mimic_iv.sqlite",
        help="Path to database for reward computation",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ehr-r1",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="Weights & Biases entity (team/username)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 precision",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2 for memory efficiency",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of generations per prompt for GRPO (must be divisible by generation_batch_size)",
    )
    parser.add_argument(
        "--reward-functions",
        nargs="+",
        default=["execution"],
        help="List of reward functions to use",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for VLLM inference",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Load environment variables
    load_dotenv()

    # Set HuggingFace cache if specified
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        hf_home = os.path.expandvars(hf_home)
        os.environ["HF_HOME"] = hf_home
        logger.info(f"HuggingFace cache directory: {hf_home}")

    # Set WandB API key if specified
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        import wandb
        wandb.login(key=wandb_api_key)
        logger.info("WandB login successful")

    # Set PyTorch CUDA memory allocation config
    pytorch_cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
    if pytorch_cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
        logger.info(f"PyTorch CUDA allocation config: {pytorch_cuda_alloc_conf}")

    # Set temporary directory
    tmpdir = os.getenv("TMPDIR")
    if tmpdir:
        tmpdir = os.path.expandvars(tmpdir)
        os.environ["TMPDIR"] = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        logger.info(f"Temporary directory: {tmpdir}")

    # Set Triton cache directory (for DeepSpeed)
    triton_cache_dir = os.getenv("TRITON_CACHE_DIR")
    if triton_cache_dir:
        triton_cache_dir = os.path.expandvars(triton_cache_dir)
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
        os.makedirs(triton_cache_dir, exist_ok=True)
        logger.info(f"Triton cache directory: {triton_cache_dir}")

    # Set Torch extensions cache directory (for DeepSpeed)
    torch_extensions_dir = os.getenv("TORCH_EXTENSIONS_DIR")
    if torch_extensions_dir:
        torch_extensions_dir = os.path.expandvars(torch_extensions_dir)
        os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir
        os.makedirs(torch_extensions_dir, exist_ok=True)
        logger.info(f"Torch extensions cache directory: {torch_extensions_dir}")

    args = parse_args()

    logger.info("Starting EHR-R1 GRPO training")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer with all parameters
    trainer = EHRSQLGRPOTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grpo_epochs=args.num_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        use_wandb=args.use_wandb,
        db_path=args.db_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        bf16=args.bf16,
        reward_functions=args.reward_functions,
        use_flash_attention=args.use_flash_attention,
        num_generations=args.num_generations,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        output_dir=args.output_dir,
        save_steps=500,
        eval_steps=500,
    )

    # Setup models first
    logger.info("Setting up models...")
    trainer.setup_models()

    # Load training dataset
    logger.info("Loading training dataset...")
    dataset = EHRSQLDataset(
        data_path=args.data_path,
        tokenizer=trainer.tokenizer,
        max_length=args.max_length,
    )

    logger.info(f"Loaded {len(dataset)} training examples")

    # Load evaluation dataset
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        logger.info("Loading evaluation dataset...")
        eval_dataset = EHRSQLDataset(
            data_path=args.eval_data_path,
            tokenizer=trainer.tokenizer,
            max_length=args.max_length,
        )
        logger.info(f"Loaded {len(eval_dataset)} evaluation examples")
    else:
        logger.warning(f"Evaluation dataset not found at {args.eval_data_path}")

    # Setup trainer with dataset
    logger.info("Setting up trainer...")
    trainer.setup_trainer(dataset, eval_dataset)

    # Note: Wandb is now initialized in the trainer itself

    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Cleanup is handled by the trainer
        pass


if __name__ == "__main__":
    main()
