"""Main evaluation script."""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ehr_r1.evaluation.evaluator import EHRSQLEvaluator
from ehr_r1.utils.logger import get_logger
from ehr_r1.utils.prompts import format_prompt
from ehr_r1.utils.schema import get_schema

logger = get_logger(__name__)


def create_output_structure(output_dir: str, model_name: str) -> dict:
    """Create organized output directory structure.

    Args:
        output_dir: Base output directory
        model_name: Model name for organizing results

    Returns:
        Dictionary with paths for different output files
    """
    # Extract model name from path (e.g., "MPX0222forHF/SQL-R1-3B" -> "SQL-R1-3B")
    model_short_name = model_name.split("/")[-1]

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure
    model_dir = Path(output_dir) / model_short_name
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths (simplified)
    paths = {
        "run_dir": str(run_dir),
        "results": str(run_dir / "results.json"),
        "log": str(run_dir / "evaluation.log"),
    }

    return paths


def setup_logging(log_path: str):
    """Setup logging to file and console."""
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logging.getLogger().addHandler(file_handler)


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
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results",
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
        choices=[
            "omnisql_prompt.jinja2",
            "simple_sql_prompt.jinja2",
            "few_shot_prompt.jinja2",
        ],
        help="Prompt template to use",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/mimic_iv/mimic_iv.sqlite",
        help="Path to the database for execution accuracy evaluation",
    )
    parser.add_argument(
        "--current_time",
        type=str,
        default="2100-12-31 23:59:00",
        help="Current time for temporal context in prompts (e.g., '2023-01-01 00:00:00')",
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

        # Remove SQL comments (lines starting with --)
        lines = last_sql.split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("--"):
                sql_lines.append(line)

        return "\n".join(sql_lines)
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
    # Load environment variables
    load_dotenv()

    # Set HuggingFace cache directory if specified
    if os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = os.getenv("HF_HOME")
        logger.info(f"HuggingFace cache directory: {os.getenv('HF_HOME')}")

    args = parse_args()

    # Create output directory structure
    output_paths = create_output_structure(args.output_dir, args.model_name)

    # Setup logging
    setup_logging(output_paths["log"])

    logger.info(f"Starting evaluation with model: {args.model_name}")
    logger.info(f"Output directory: {output_paths['run_dir']}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max length: {args.max_length}")

    # Prepare configuration for later inclusion in results
    config = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "tensor_parallel_size": args.tensor_parallel_size,
        "temperature": args.temperature,
        "prompt_template": args.prompt_template,
        "db_path": args.db_path,
        "timestamp": datetime.now().isoformat(),
    }

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

    # Get VLLM settings from environment
    vllm_cache_dir = os.getenv("VLLM_CACHE_ROOT", "")
    gpu_memory_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.92"))

    # Set VLLM cache directory
    os.environ["VLLM_CACHE_ROOT"] = vllm_cache_dir
    logger.info(f"VLLM cache directory: {vllm_cache_dir}")

    # Initialize VLLM
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_util,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        download_dir=vllm_cache_dir,
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

        # Load corresponding labels (SQL queries)
        labels_path = args.dataset_path.replace("data.json", "label.json")
        logger.info(f"Loading labels from: {labels_path}")

        with open(labels_path, "r") as f:
            labels = json.load(f)

    except Exception as e:
        logger.error(f"Could not load dataset or labels. Error: {e}")
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
            example_id = example.get("id", "")
            question = example.get("question", "")
            # Get database schema - use from example or default to MIMIC-IV
            db_name = example.get("db_id", example.get("database", "mimic_iv"))
            schema = get_schema(db_name)
            if not schema:
                logger.warning(
                    f"Schema not found for database: {db_name}, using MIMIC-IV"
                )
                schema = (
                    get_schema("mimic_iv") or "mimic_iv"
                )  # Fallback to string if still None

            # Get target SQL from labels using the example ID
            target_sql = labels.get(example_id, "")

            if not question:
                logger.warning(f"Skipping sample {i}: missing question")
                continue

            if (
                not target_sql or target_sql.strip().lower() == "null"
            ):  # TODO: Need to evaluate "Unanswerable" cases
                logger.warning(
                    f"Skipping sample {i}: missing or null target SQL for ID {example_id}"
                )
                continue

            # Format prompt using template
            prompt = format_prompt(
                template_name=args.prompt_template,
                schema=schema,
                question=question,
                current_time=args.current_time,
            )

            # Apply chat template
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            chat_prompts.append(chat_prompt)
            targets.append(target_sql)

        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue

    logger.info(f"Generating responses for {len(chat_prompts)} samples...")

    # Generate responses
    outputs = llm.generate(chat_prompts, sampling_params)

    # Parse predictions
    predictions = []
    sample_results = []

    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        pred_sql = parse_response(response)
        predictions.append(pred_sql)

        # Save sample details for inclusion in final results
        if i < len(targets):
            sample_detail = {
                "sample_id": i,
                "predicted_sql": pred_sql,
                "target_sql": targets[i],
                "raw_response": response,
            }
            sample_results.append(sample_detail)

    # Print sample for debugging
    for i in range(min(3, len(predictions))):
        logger.info(f"\nSample {i}:")
        logger.info(f"Target: {targets[i]}")
        logger.info(f"Prediction: {predictions[i]}")
        logger.info("-" * 50)

    # Compute metrics using existing evaluator
    if predictions and targets:
        # Initialize evaluator with database path for execution accuracy
        evaluator = EHRSQLEvaluator(db_path=args.db_path)
        results = evaluator.evaluate(predictions, targets, output_paths["results"])

        # Add execution details to sample results
        if "detailed_results" in results and len(results["detailed_results"]) == len(
            sample_results
        ):
            for i, execution_detail in enumerate(results["detailed_results"]):
                if i < len(sample_results):
                    sample_results[i].update(
                        {
                            "predicted_execution_success": execution_detail.get(
                                "predicted_success", False
                            ),
                            "predicted_execution_result": execution_detail.get(
                                "predicted_result", None
                            ),
                            "predicted_execution_error": execution_detail.get(
                                "predicted_error", None
                            ),
                            "target_execution_success": execution_detail.get(
                                "ground_truth_success", False
                            ),
                            "target_execution_result": execution_detail.get(
                                "ground_truth_result", None
                            ),
                            "target_execution_error": execution_detail.get(
                                "ground_truth_error", None
                            ),
                            "execution_match": execution_detail.get(
                                "execution_match", False
                            ),
                            "exact_match": execution_detail.get("exact_match", False),
                        }
                    )

        # Create comprehensive results
        final_results = {
            "config": config,
            "evaluation_metrics": {
                "exact_match_accuracy": results.get("exact_match_accuracy", 0),
                "execution_accuracy": results.get("execution_accuracy", 0),
                "predicted_success_rate": results.get("predicted_success_rate", 0),
                "ground_truth_success_rate": results.get(
                    "ground_truth_success_rate", 0
                ),
                "total_predictions": results.get("total_predictions", 0),
                "non_empty_predictions": results.get("non_empty_predictions", 0),
            },
            "sample_results": sample_results,
        }

        # Save comprehensive results
        with open(output_paths["results"], "w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"\n=== EVALUATION RESULTS ===")
        logger.info(
            f"Exact Match Accuracy: {results.get('exact_match_accuracy', 'N/A'):.4f}"
        )
        if "execution_accuracy" in results:
            logger.info(
                f"Execution Accuracy: {results.get('execution_accuracy', 'N/A'):.4f}"
            )
            logger.info(
                f"Predicted Success Rate: {results.get('predicted_success_rate', 'N/A'):.4f}"
            )
            logger.info(
                f"Ground Truth Success Rate: {results.get('ground_truth_success_rate', 'N/A'):.4f}"
            )
        logger.info(f"Results saved to: {output_paths['results']}")

        # Clean up any temporary detailed files created by evaluator
        temp_detailed_path = output_paths["results"].replace(".json", "_detailed.json")
        if os.path.exists(temp_detailed_path):
            os.remove(temp_detailed_path)

        # Final summary
        logger.info(f"\n=== EVALUATION SUMMARY ===")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Samples evaluated: {len(predictions)}")
        logger.info(f"All results saved in: {output_paths['run_dir']}")

    else:
        logger.error("No valid predictions generated!")


if __name__ == "__main__":
    main()
