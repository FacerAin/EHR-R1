"""Main evaluation script."""

import argparse
import json
from typing import Optional

from ehr_r1.data.ehrsql_dataset import EHRSQLDataset
from ehr_r1.training.grpo_trainer import EHRSQLGRPOTrainer
from ehr_r1.evaluation.evaluator import EHRSQLEvaluator
from ehr_r1.utils.config import ModelConfig, DataConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate EHR-R1 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Initialize configurations
    data_config = DataConfig(
        test_data_path=args.test_data_path,
    )
    
    # Load test dataset
    test_dataset = EHRSQLDataset(
        data_path=data_config.test_data_path,
        split="test",
        max_length=data_config.max_length,
    )
    
    # Load trained model
    trainer = EHRSQLGRPOTrainer()
    trainer.setup_models()
    
    # Initialize evaluator
    evaluator = EHRSQLEvaluator()
    
    # Run evaluation
    predictions = []
    ground_truths = []
    schemas = []
    expected_results = []
    
    # TODO: Implement evaluation loop
    
    # Compute metrics
    results = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
    )
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {args.output_path}")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()