"""Main evaluation script."""

import argparse
import json
import re
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ehr_r1.evaluation.evaluator import EHRSQLEvaluator
from ehr_r1.utils.logger import get_logger
from ehr_r1.utils.schema import get_schema
from ehr_r1.utils.prompts import format_sql_prompt

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate EHRSQL model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="MPX0222forHF/SQL-R1-3B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/mimic_iv/test/data.json",
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max generation length",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="omnisql_prompt.jinja2",
        choices=["omnisql_prompt.jinja2", "simple_sql_prompt.jinja2", "few_shot_prompt.jinja2"],
        help="Prompt template to use",
    )
    return parser.parse_args()


def clean_sql(sql: str) -> str:
    """Clean and normalize SQL query."""
    sql = re.sub(r"\s+", " ", sql.strip())
    sql = sql.rstrip(";")
    return sql


def parse_response(response: str) -> str:
    """Parse SQL response from model output."""
    pattern = r"```sql\s*(.*?)\s*```"

    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        return ""


def get_stop_token_ids(model_name: str) -> list:
    """Get stop token IDs based on model name."""
    if "Qwen2.5-" in model_name:
        return [151645]  # <|im_end|>
    elif "deepseek-coder-" in model_name:
        return [32021]
    elif "DeepSeek-Coder-V2" in model_name:
        return [100001]
    elif "OpenCoder-" in model_name:
        return [96539]
    elif "Meta-Llama-" in model_name:
        return [128009, 128001]
    elif "granite-" in model_name:
        return [0]  # <|end_of_text|>
    elif "starcoder2-" in model_name:
        return [0]  # <|end_of_text|>
    elif "Codestral-" in model_name:
        return [2]
    elif "Mixtral-" in model_name:
        return [2]
    elif "OmniSQL-" in model_name:
        return [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
    else:
        print("Use Qwen2.5's stop tokens by default.")
        return [151645]


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    logger.info(f"Starting evaluation with model: {args.model_name}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max length: {args.max_length}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Get stop token IDs
    stop_token_ids = get_stop_token_ids(args.model_name)
    logger.info(f"Stop token IDs: {stop_token_ids}")

    # Set up VLLM parameters
    max_model_len = 8192
    max_output_len = args.max_length

    logger.info(f"Max model length: {max_model_len}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=max_output_len,
        n=1,
        stop_token_ids=stop_token_ids,
    )

    # Initialize VLLM
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        swap_space=42,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
    )

    logger.info(f"Loading dataset: {args.dataset_path}")

    # Load dataset from JSON file
    try:
        with open(args.dataset_path, "r") as f:
            data = json.load(f)

        # Extract data array from JSON structure
        if isinstance(data, dict) and "data" in data:
            dataset = data["data"]
        else:
            dataset = data

    except Exception as e:
        logger.error(f"Could not load {args.dataset_path}. Error: {e}")
        return

    # Limit samples if specified
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset[: args.num_samples]

    logger.info(f"Evaluating on {len(dataset)} samples...")

    # Prepare prompts
    chat_prompts = []
    targets = []

    for i, example in enumerate(tqdm(dataset, desc="Preparing prompts")):
        try:
            # Extract fields from EHRSQL dataset structure
            question = example.get("question", "")
            # Get database schema - use from example or default to MIMIC-IV
            db_name = example.get("db_id", example.get("database", "mimic_iv"))
            schema = get_schema(db_name)
            if not schema:
                logger.warning(f"Schema not found for database: {db_name}, using MIMIC-IV")
                schema = get_schema("mimic_iv") or "mimic_iv"  # Fallback to string if still None
            
            target_sql = example.get("query", example.get("sql", ""))

            if not question:
                logger.warning(f"Skipping sample {i}: missing question")
                continue

            # Format prompt using template
            prompt = format_sql_prompt(schema=schema, question=question, template=args.prompt_template)

            # Apply chat template
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            chat_prompts.append(chat_prompt)
            targets.append(target_sql if target_sql else "")

        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue

    logger.info(f"Generating responses for {len(chat_prompts)} samples...")

    # Generate responses
    outputs = llm.generate(chat_prompts, sampling_params)

    # Parse predictions
    predictions = []
    for output in outputs:
        response = output.outputs[0].text
        pred_sql = parse_response(response)
        predictions.append(pred_sql)

    # Print sample for debugging
    for i in range(min(3, len(predictions))):
        logger.info(f"\nSample {i}:")
        logger.info(f"Target: {targets[i]}")
        logger.info(f"Prediction: {predictions[i]}")
        logger.info("-" * 50)

    # Compute metrics using existing evaluator
    if predictions and targets:
        evaluator = EHRSQLEvaluator()
        results = evaluator.evaluate(predictions, targets, args.output_path)

        # Add model info to results
        results.update(
            {
                "model_name": args.model_name,
                "dataset_path": args.dataset_path,
            }
        )

        logger.info(f"\n=== EVALUATION RESULTS ===")
        logger.info(
            f"Exact Match Accuracy: {results.get('exact_match_accuracy', 'N/A'):.4f}"
        )
        logger.info(f"Results saved to: {args.output_path}")

    else:
        logger.error("No valid predictions generated!")


if __name__ == "__main__":
    main()
