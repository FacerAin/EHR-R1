"""Reward model for EHRSQL evaluation."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..utils.sql_executor import SQLExecutor


class EHRSQLRewardModel:
    """Simple reward model based on SQL execution success/failure."""

    def __init__(
        self,
        db_path: str,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        execution_match_bonus: float = 2.0,
    ):
        self.db_path = db_path
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.execution_match_bonus = execution_match_bonus
        self.sql_executor = SQLExecutor(db_path)

    def connect(self):
        """Connect to database."""
        return self.sql_executor.connect()

    def disconnect(self):
        """Disconnect from database."""
        self.sql_executor.disconnect()

    def compute_reward(
        self,
        predicted_sql: str,
        target_sql: str,
        question: str = "",
        schema: str = "",
    ) -> float:
        """
        Compute reward for a predicted SQL query.

        Args:
            predicted_sql: Generated SQL query
            target_sql: Ground truth SQL query
            question: Original question (unused for now)
            schema: Database schema (unused for now)

        Returns:
            Reward score
        """
        if not predicted_sql.strip():
            return self.failure_reward

        # Execute predicted SQL
        pred_success, pred_result, pred_error = self.sql_executor.execute_query(
            predicted_sql
        )

        if not pred_success:
            # SQL execution failed - negative reward
            return self.failure_reward

        # SQL executed successfully - positive reward
        reward = self.success_reward

        # Check if we have ground truth to compare against
        if target_sql.strip():
            target_success, target_result, target_error = (
                self.sql_executor.execute_query(target_sql)
            )

            if target_success and pred_result == target_result:
                # Results match - bonus reward
                reward += self.execution_match_bonus

        return reward

    def compute_batch_rewards(
        self,
        predicted_sqls: List[str],
        target_sqls: List[str],
        questions: Optional[List[str]] = None,
        schemas: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute rewards for a batch of predictions."""
        rewards = []

        for i, (pred_sql, target_sql) in enumerate(zip(predicted_sqls, target_sqls)):
            question = questions[i] if questions else ""
            schema = schemas[i] if schemas else ""

            reward = self.compute_reward(pred_sql, target_sql, question, schema)
            rewards.append(reward)

        return rewards


class EHRSQLNeuralRewardModel(nn.Module):
    """Neural reward model for more sophisticated reward computation (future enhancement)."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward pass of the reward model."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def compute_reward(
        self,
        query: str,
        predicted_sql: str,
        target_sql: str,
        tokenizer: AutoTokenizer,
    ) -> float:
        """Compute reward using neural model (placeholder for future enhancement)."""
        # TODO: Implement neural reward computation
        return 0.0
