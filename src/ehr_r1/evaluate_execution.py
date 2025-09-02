"""Standalone execution accuracy evaluation script."""

import argparse
import json
from pathlib import Path

from ehr_r1.evaluation.evaluator import EHRSQLEvaluator


def main():
    """Main execution accuracy evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate EHRSQL execution accuracy")

    parser.add_argument(
        "--predictions_file", required=True, help="JSON file with predicted SQL queries"
    )
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="JSON file with ground truth SQL queries",
    )
    parser.add_argument("--db_path", required=True, help="Path to SQLite database file")
    parser.add_argument(
        "--output_file",
        default="execution_evaluation_results.json",
        help="Output file for evaluation results",
    )

    args = parser.parse_args()

    # Validate files exist
    for file_path in [args.predictions_file, args.ground_truth_file, args.db_path]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    print("Loading evaluation data...")

    # Load predictions
    with open(args.predictions_file, "r") as f:
        pred_data = json.load(f)
    print(f"Loaded {len(pred_data)} predictions")

    # Load ground truths
    with open(args.ground_truth_file, "r") as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} ground truth queries")

    # Extract SQL queries
    predictions = []
    ground_truths = []

    # Create lookup for ground truth by id
    gt_lookup = {item.get("id", i): item for i, item in enumerate(gt_data)}

    matched_pairs = 0
    for i, pred_item in enumerate(pred_data):
        pred_id = pred_item.get("id", i)

        if pred_id in gt_lookup:
            gt_item = gt_lookup[pred_id]
            predictions.append(pred_item.get("predicted_sql", ""))
            ground_truths.append(gt_item.get("query", ""))
            matched_pairs += 1
        else:
            print(f"Warning: No ground truth found for prediction ID {pred_id}")

    print(f"Matched {matched_pairs} prediction-ground truth pairs")

    if matched_pairs == 0:
        print("Error: No matching pairs found!")
        return

    # Initialize evaluator
    print(f"Connecting to database: {args.db_path}")
    evaluator = EHRSQLEvaluator(db_path=args.db_path)

    try:
        # Run evaluation
        print("Starting execution accuracy evaluation...")
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truths=ground_truths,
            output_file=args.output_file,
        )

        # Print summary results
        print("\n" + "=" * 50)
        print("EXECUTION ACCURACY EVALUATION RESULTS")
        print("=" * 50)

        print(f"Total predictions evaluated: {results['total_predictions']}")
        print(f"Execution accuracy: {results.get('execution_accuracy', 'N/A'):.4f}")
        print(f"Exact match accuracy: {results['exact_match_accuracy']:.4f}")
        print(
            f"Predicted success rate: {results.get('predicted_success_rate', 'N/A'):.4f}"
        )
        print(
            f"Ground truth success rate: {results.get('ground_truth_success_rate', 'N/A'):.4f}"
        )
        print(f"Empty prediction rate: {results['empty_prediction_rate']:.4f}")

        # Component accuracy
        print(f"Keyword match accuracy: {results.get('keyword_match', 'N/A'):.4f}")

        print(f"\nDetailed results saved to: {args.output_file}")

        if "execution_accuracy" in results:
            detailed_file = args.output_file.replace(".json", "_detailed.json")
            print(f"Detailed execution results saved to: {detailed_file}")

    finally:
        # Clean up
        evaluator.close()


if __name__ == "__main__":
    main()
