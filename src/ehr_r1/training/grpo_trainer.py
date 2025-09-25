"""GRPO training implementation using TRL."""

from typing import List, Optional

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
import os

from ..models import reward_model
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EHRSQLGRPOTrainer:
    """GRPO trainer for EHRSQL task using TRL."""

    def __init__(
        self,
        model_name: str = "MPX0222forHF/SQL-R1-3B",
        learning_rate: float = 1e-5,
        bf16: bool = True,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        grpo_epochs: int = 1,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        use_wandb: bool = True,
        beta: float = 0.1,
        db_path: str = "data/mimic_iv/mimic_iv.sqlite",
        wandb_project: str = "ehr-r1",
        wandb_run_name: Optional[str] = None,
        reward_functions: Optional[List[str]] = None,
        num_generations: int = 4,
        use_flash_attention: bool = False,
        vllm_tensor_parallel_size: int = 4,
    ):
        self.model_name = model_name
        self.db_path = db_path
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.use_flash_attention = use_flash_attention
        self.num_generations = num_generations
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size

        # Setup reward functions
        if reward_functions is None:
            reward_functions = ["execution"]
        self.reward_functions = [
            reward_model.AVAILABLE_REWARD_FUNCTIONS[name] 
            for name in reward_functions 
            if name in reward_model.AVAILABLE_REWARD_FUNCTIONS
        ]

        # Validate length parameters
        completion_length = max_length - max_prompt_length
        if completion_length <= 0:
            logger.warning(f"Invalid completion length: max_length={max_length}, max_prompt_length={max_prompt_length}")
            completion_length = 256
            logger.info(f"Setting min completion_length to {completion_length}")
        
        # Initialize GRPO configuration
        self.config = GRPOConfig(
            bf16=bf16,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=grpo_epochs,
            max_prompt_length=max_prompt_length,
            max_completion_length=max(completion_length, 256),
            report_to="wandb" if use_wandb else None,
            beta=beta,
            save_steps=50,
            eval_steps=50,
            logging_steps=10,
            remove_unused_columns=False,
            output_dir="./outputs",
            do_train=True,
            do_eval=True,  # Enable evaluation
            num_generations=self.num_generations,  # Must be divisible by generation_batch_size
            use_vllm=True,  # TODO: make this configurable
            vllm_mode="colocate",
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
        )

        self.tokenizer = None
        self.model = None
        self.grpo_trainer = None

        # Initialize wandb if enabled (only on main process)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.use_wandb and local_rank == 0:
            self._setup_wandb()

    def setup_models(self):
        """Initialize models and tokenizer."""
        logger.info(f"Loading tokenizer and model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

                
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        # Add flash attention if requested
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        # Load main model for training
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()

        # Setup SQL executor for reward functions
        try:
            reward_model.setup_sql_executor(self.db_path)
        except Exception as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise RuntimeError(f"Database connection failed during reward model setup: {e}") from e
        
        logger.info("Models loaded successfully")

    def setup_trainer(self, dataset, eval_dataset):
        """Setup GRPO trainer with dataset."""
        if not self.model or not self.tokenizer:
            raise ValueError("Models not initialized. Call setup_models() first.")

        logger.info("Setting up GRPO trainer")

        # Create GRPO trainer with multiple reward functions
        trainer_kwargs = {
            "model": self.model,
            "reward_funcs": self.reward_functions,
            "args": self.config,
            "train_dataset": dataset,
            "eval_dataset": eval_dataset,
            "processing_class": self.tokenizer,
        }
        
        self.grpo_trainer = GRPOTrainer(**trainer_kwargs)
        logger.info("GRPO trainer setup complete")

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            # Generate descriptive run name if not provided
            if not self.wandb_run_name:
                import datetime
                model_short = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
                reward_funcs = "-".join([func.__name__ for func in self.reward_functions])
                timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
                self.wandb_run_name = f"{model_short}_lr{self.config.learning_rate}_beta{self.config.beta}_{reward_funcs}_{timestamp}"

            # Initialize wandb
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "model_name": self.model_name,
                    "learning_rate": self.config.learning_rate,
                    "per_device_train_batch_size": self.config.per_device_train_batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "num_train_epochs": self.config.num_train_epochs,
                    "max_prompt_length": self.config.max_prompt_length,
                    "beta": self.config.beta,
                    "reward_functions": [func.__name__ for func in self.reward_functions],
                    "db_path": self.db_path,
                    "num_generations": self.num_generations,
                },
            )

            logger.info(f"Initialized Weights & Biases project: {self.wandb_project}")

        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.use_wandb = False


    def train(
        self,
        num_epochs: int = 1,
        save_steps: int = 500,
        eval_steps: int = 500,
        output_dir: str = "./outputs",
    ):
        """Main training loop."""
        if not self.grpo_trainer:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        logger.info(f"Starting GRPO training for {num_epochs} epochs")
        logger.info(f"Output directory: {output_dir}")

        # Update config with new parameters
        self.grpo_trainer.args.output_dir = output_dir
        self.grpo_trainer.args.num_train_epochs = num_epochs
        self.grpo_trainer.args.save_steps = save_steps
        self.grpo_trainer.args.eval_steps = eval_steps

        # Start training
        try:
            self.grpo_trainer.train()
            logger.info("Training completed successfully")

            # Save final model
            final_output_dir = f"{output_dir}/final"
            self.grpo_trainer.save_model(final_output_dir)
            logger.info(f"Final model saved to {final_output_dir}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            # Clean up SQL executor
            reward_model.cleanup_sql_executor()
            # Finish wandb run (only on main process)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if self.use_wandb and local_rank == 0:
                try:
                    import wandb
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")
