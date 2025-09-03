"""GRPO training implementation using TRL."""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from ..models.reward_model import EHRSQLRewardModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EHRSQLGRPOTrainer:
    """GRPO trainer for EHRSQL task using TRL."""

    def __init__(
        self,
        model_name: str = "MPX0222forHF/SQL-R1-3B",
        learning_rate: float = 1e-5,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        grpo_epochs: int = 1,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        use_wandb: bool = True,
        beta: float = 0.1,
        db_path: str = "data/mimic_iv/mimic_iv.sqlite",
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
        reward_match_bonus: float = 2.0,
        wandb_project: str = "ehr-r1",
        wandb_run_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.db_path = db_path
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        # Initialize GRPO configuration
        self.config = GRPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=grpo_epochs,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            log_with="wandb" if use_wandb else None,
            beta=beta,
            save_steps=500,
            eval_steps=500,
            logging_steps=10,
            remove_unused_columns=False,
        )

        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.grpo_trainer = None
        self.reward_model = None

        # Reward model parameters
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.reward_match_bonus = reward_match_bonus

        # Initialize wandb if enabled
        if self.use_wandb:
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Load reference model (for GRPO)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Initialize reward model
        try:
            self.reward_model.connect()
        except Exception as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise RuntimeError(f"Database connection failed during reward model setup: {e}") from e
        
        logger.info("Models loaded successfully")

    def setup_trainer(self, dataset):
        """Setup GRPO trainer with dataset."""
        if not self.model or not self.tokenizer:
            raise ValueError("Models not initialized. Call setup_models() first.")

        logger.info("Setting up GRPO trainer")

        # Create GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            config=self.config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
        )

        logger.info("GRPO trainer setup complete")

    def parse_sql_response(self, response: str) -> str:
        """Parse SQL response from model output."""
        # Look for SQL code blocks
        pattern = r"```sql\s*(.*?)\s*```"
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            # Extract the last SQL query and clean it
            sql = sql_blocks[-1].strip()

            # Remove SQL comments (lines starting with --)
            lines = sql.split("\n")
            sql_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("--"):
                    sql_lines.append(line)

            return "\n".join(sql_lines)

        return response.strip()

    def compute_rewards(
        self, queries: List[str], responses: List[str], target_sqls: List[str]
    ) -> List[float]:
        """Compute rewards for generated responses."""
        rewards = []

        for query, response, target_sql in zip(queries, responses, target_sqls):
            # Parse SQL from response
            predicted_sql = self.parse_sql_response(response)

            # Compute reward using reward model
            reward = self.reward_model.compute_reward(
                predicted_sql=predicted_sql,
                target_sql=target_sql,
                question=query,
            )

            rewards.append(reward)

        return rewards

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
                    "max_length": self.config.max_length,
                    "max_prompt_length": self.config.max_prompt_length,
                    "beta": self.config.beta,
                    "reward_success": self.reward_success,
                    "reward_failure": self.reward_failure,
                    "reward_match_bonus": self.reward_match_bonus,
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

    def _log_training_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics to wandb."""
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to wandb: {e}")

    def _log_reward_distribution(self, rewards: List[float], step: int):
        """Log reward distribution to wandb."""
        if self.use_wandb and rewards:
            try:
                wandb.log(
                    {
                        "reward/mean": np.mean(rewards),
                        "reward/std": np.std(rewards),
                        "reward/min": np.min(rewards),
                        "reward/max": np.max(rewards),
                        "reward/median": np.median(rewards),
                        "reward/positive_ratio": sum(1 for r in rewards if r > 0)
                        / len(rewards),
                    },
                    step=step,
                )

                # Log reward histogram
                wandb.log({"reward/histogram": wandb.Histogram(rewards)}, step=step)

            except Exception as e:
                logger.warning(f"Failed to log reward distribution to wandb: {e}")

    def train(
        self,
        dataset,
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
            # Clean up reward model connection
            if self.reward_model:
                self.reward_model.disconnect()

            # Finish wandb run
            if self.use_wandb:
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
