"""GRPO training implementation using TRL."""

from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from ..models import reward_model
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GenerationLoggingCallback(TrainerCallback):
    """Callback to log generation examples during training."""
    
    def __init__(self, trainer_instance, log_every_n_steps: int = 10):
        self.trainer_instance = trainer_instance
        self.log_every_n_steps = log_every_n_steps
    
    def on_log(self, args, state, control, model=None, **kwargs):
        """Called when logging occurs."""
        if state.global_step % self.log_every_n_steps == 0:
            self.trainer_instance._log_generation_examples(state.global_step)


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
    ):
        self.model_name = model_name
        self.db_path = db_path
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

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
            save_steps=500,
            eval_steps=500,
            logging_steps=10,
            remove_unused_columns=False,
            output_dir="./outputs",
            do_train=True,
            do_eval=True,  # Enable evaluation
            num_generations=4,  # Must be divisible by generation_batch_size
        )

        self.tokenizer = None
        self.model = None
        self.grpo_trainer = None

        # Initialize wandb if enabled (only on main process)
        import os
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

        # Load main model for training
        # Don't use device_map in distributed training
        # import os
        # is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
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
        
        # Add generation logging callback
        if self.use_wandb:
            generation_callback = GenerationLoggingCallback(self, log_every_n_steps=10)
            self.grpo_trainer.add_callback(generation_callback)
            logger.info("Added generation logging callback (every 10 steps)")

        logger.info("GRPO trainer setup complete")

    def _log_generation_examples(self, step: int, num_examples: int = 3):
        """Log generation examples to WandB for qualitative evaluation."""
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not self.use_wandb or local_rank != 0:
            return
            
        try:
            # Get some sample prompts from the dataset
            if hasattr(self.grpo_trainer, 'train_dataset') and self.grpo_trainer.train_dataset:
                dataset = self.grpo_trainer.train_dataset
                # Sample a few examples
                import random
                sample_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
                
                examples_table = []
                for idx in sample_indices:
                    sample = dataset[idx]
                    # Get the prompt (input text before generation)
                    prompt = self.tokenizer.decode(sample['input_ids'][:sample.get('prompt_length', len(sample['input_ids'])//2)], skip_special_tokens=True)
                    
                    # Generate response using current model
                    try:
                        response = self.generate_response(
                            prompt, 
                            max_new_tokens=200, 
                            temperature=0.1
                        )
                        
                        # Try to get ground truth if available
                        ground_truth = sample.get('target_sql', 'N/A')
                        if isinstance(ground_truth, list) and ground_truth:
                            ground_truth = ground_truth[0]
                        
                        examples_table.append({
                            "step": step,
                            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                            "generated_sql": response,
                            "ground_truth": ground_truth[:500] + "..." if isinstance(ground_truth, str) and len(ground_truth) > 500 else str(ground_truth)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to generate example {idx}: {e}")
                        continue
                
                if examples_table:
                    wandb.log({
                        "generation_examples": wandb.Table(
                            columns=["step", "prompt", "generated_sql", "ground_truth"],
                            data=[[ex["step"], ex["prompt"], ex["generated_sql"], ex["ground_truth"]] for ex in examples_table]
                        )
                    }, step=step)
                    
                    logger.info(f"Logged {len(examples_table)} generation examples to WandB")
                    
        except Exception as e:
            logger.warning(f"Failed to log generation examples: {e}")

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
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
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if self.use_wandb and local_rank == 0:
                try:
                    import wandb
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 0.1,
    ) -> str:
        """Generate SQL response for given query."""
        if not self.model or not self.tokenizer:
            raise ValueError("Models not initialized. Call setup_models() first.")

        # Tokenize input
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()
