"""Reward functions for EHRSQL evaluation."""

import re
from typing import List

from ..utils.sql_executor import SQLExecutor

# Global SQL executor instance (initialized by setup_sql_executor)
_sql_executor = None


def setup_sql_executor(db_path: str):
    """Initialize the global SQL executor."""
    global _sql_executor
    _sql_executor = SQLExecutor(db_path)
    _sql_executor.connect()


def cleanup_sql_executor():
    """Cleanup the global SQL executor."""
    global _sql_executor
    if _sql_executor:
        _sql_executor.disconnect()
        _sql_executor = None


def parse_sql_response(response: str) -> str:
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


def execution_reward_func(completions, target_sqls, **_) -> List[float]:
    """GRPO-compatible reward function for SQL execution success."""
    if not _sql_executor:
        raise RuntimeError("SQL executor not initialized. Call setup_sql_executor() first.")
    
    # Extract SQL from completions
    predicted_sqls = [parse_sql_response(completion) for completion in completions]
    
    rewards = []
    for pred_sql, target_sql in zip(predicted_sqls, target_sqls):
        if not pred_sql.strip():
            rewards.append(-1.0)
            continue

        # Execute predicted SQL
        pred_success, pred_result, _ = _sql_executor.execute_query(pred_sql)

        if not pred_success:
            rewards.append(-1.0)
            continue

        # SQL executed successfully - base reward
        reward = 1.0

        # Check if results match ground truth
        if target_sql.strip():
            target_success, target_result, _ = _sql_executor.execute_query(target_sql)
            if target_success and pred_result == target_result:
                reward += 2.0  # Match bonus

        rewards.append(reward)
    
    return rewards


def sql_syntax_reward_func(completions, **_) -> List[float]:
    """Reward function for valid SQL syntax."""
    if not _sql_executor:
        raise RuntimeError("SQL executor not initialized. Call setup_sql_executor() first.")
    
    # Extract SQL from completions
    predicted_sqls = [parse_sql_response(completion) for completion in completions]
    
    rewards = []
    for pred_sql in predicted_sqls:
        if not pred_sql.strip():
            rewards.append(-0.5)
            continue

        # Check if SQL can be executed (syntax is valid)
        pred_success, _, _ = _sql_executor.execute_query(pred_sql)
        rewards.append(0.5 if pred_success else -0.5)
    
    return rewards


def sql_format_reward_func(completions, **_) -> List[float]:
    """Reward function for proper SQL formatting (has SQL code block)."""
    rewards = []
    for completion in completions:
        # Check if response contains SQL code block
        if "```sql" in completion and "```" in completion:
            rewards.append(0.2)
        else:
            rewards.append(-0.2)
    
    return rewards


# List of available reward functions for easy configuration
AVAILABLE_REWARD_FUNCTIONS = {
    "execution": execution_reward_func,
    "syntax": sql_syntax_reward_func, 
    "format": sql_format_reward_func,
}
