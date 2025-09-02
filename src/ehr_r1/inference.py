"""Inference module for SQL generation."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.prompts import EHRSQLPromptTemplate


class EHRSQLInference:
    """Inference engine for EHRSQL text-to-SQL generation."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = self._get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = None
        self.tokenizer = None

    def _get_device(self, device: str) -> str:
        """Get appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt(self, question: str, db_details: str) -> str:
        """Create prompt for SQL generation using the standard template."""
        return EHRSQLPromptTemplate.create_prompt(question, db_details)

    def generate_sql(self, question: str, db_details: str) -> str:
        """Generate SQL query from natural language question."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        prompt = self.create_prompt(question, db_details)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Increased for longer prompt
        )

        if "cuda" in self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract SQL from the response using the template utility
        sql_query = EHRSQLPromptTemplate.extract_sql_from_response(response)

        return sql_query

    def batch_generate_sql(
        self,
        questions: List[str],
        db_details_list: List[str],
        batch_size: int = 8,
    ) -> List[str]:
        """Generate SQL queries for batch of questions."""
        results = []

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_db_details = db_details_list[i : i + batch_size]

            batch_results = []
            for question, db_details in zip(batch_questions, batch_db_details):
                sql = self.generate_sql(question, db_details)
                batch_results.append(sql)

            results.extend(batch_results)
            print(
                f"Processed {min(i + batch_size, len(questions))}/{len(questions)} samples"
            )

        return results

    def run_inference_on_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 8,
    ):
        """Run inference on a JSON file and save results."""
        print(f"Loading data from {input_file}")
        with open(input_file, "r") as f:
            data = json.load(f)

        questions = []
        db_details_list = []
        ids = []

        for item in data:
            questions.append(item["question"])
            db_details_list.append(item.get("db_details", item.get("schema", "")))
            ids.append(item.get("id", len(ids)))

        print(f"Generating SQL for {len(questions)} questions")
        predicted_sqls = self.batch_generate_sql(questions, db_details_list, batch_size)

        # Save results
        results = []
        for i, (question, db_details, predicted_sql) in enumerate(
            zip(questions, db_details_list, predicted_sqls)
        ):
            results.append(
                {
                    "id": ids[i],
                    "question": question,
                    "db_details": db_details,
                    "predicted_sql": predicted_sql,
                }
            )

        print(f"Saving results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Inference completed. Results saved to {output_file}")


def main():
    """CLI interface for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run EHRSQL inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--input_file", required=True, help="Input JSON file")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max new tokens"
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")

    args = parser.parse_args()

    inference = EHRSQLInference(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    inference.load_model()
    inference.run_inference_on_file(
        args.input_file,
        args.output_file,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
