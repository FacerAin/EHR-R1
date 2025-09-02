"""SQL execution utilities for evaluation."""

import sqlite3
import pandas as pd
from typing import Any, List, Tuple, Optional, Dict
import sqlparse
import hashlib
import json
from pathlib import Path


class SQLExecutor:
    """SQL execution engine for evaluating query results."""
    
    def __init__(self, db_path: str, timeout: int = 30):
        self.db_path = db_path
        self.timeout = timeout
        self.connection = None
        
    def connect(self):
        """Connect to the database."""
        try:
            self.connection = sqlite3.connect(self.db_path, timeout=self.timeout)
            self.connection.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout
            return True
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from the database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            
    def execute_query(self, query: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute SQL query and return results.
        
        Returns:
            Tuple of (success, result, error_message)
        """
        if not self.connection:
            if not self.connect():
                return False, None, "Failed to connect to database"
                
        try:
            # Clean and validate query
            cleaned_query = self._clean_query(query)
            if not cleaned_query:
                return False, None, "Empty or invalid query"
                
            # Execute query
            cursor = self.connection.cursor()
            cursor.execute(cleaned_query)
            
            # Fetch results
            if cleaned_query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                # Convert to list of lists for easier comparison
                result = [list(row) for row in result]
            else:
                result = cursor.rowcount
                
            return True, result, None
            
        except sqlite3.Error as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
            
    def _clean_query(self, query: str) -> str:
        """Clean and normalize SQL query."""
        if not query:
            return ""
            
        # Remove common artifacts from generation
        query = query.strip()
        
        # Remove markdown code blocks if present
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
            
        # Remove trailing semicolon if present
        query = query.rstrip(';').strip()
        
        # Basic SQL injection prevention (very basic)
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                print(f"Warning: Potentially dangerous keyword '{keyword}' found in query")
                
        return query
        
    def normalize_result(self, result: Any) -> Any:
        """Normalize query results for comparison."""
        if result is None:
            return None
            
        if isinstance(result, list):
            # Sort list of lists for consistent comparison
            if result and isinstance(result[0], list):
                # Sort each row, then sort rows
                normalized = []
                for row in result:
                    # Convert all values to string for consistent comparison
                    str_row = [str(val) if val is not None else None for val in row]
                    normalized.append(str_row)
                # Sort the entire result
                try:
                    normalized.sort()
                except TypeError as e:
                    # If sorting fails, log the error and return as is
                    print(f"Failed to sort normalized result due to TypeError: {e}")
                return normalized
            else:
                # Simple list, sort if possible
                try:
                    return sorted([str(val) if val is not None else None for val in result])
                except TypeError:
                    return [str(val) if val is not None else None for val in result]
        else:
            return str(result) if result is not None else None
            
    def compare_results(self, result1: Any, result2: Any) -> bool:
        """Compare two query results for equality."""
        norm1 = self.normalize_result(result1)
        norm2 = self.normalize_result(result2)
        
        return norm1 == norm2
        
    def get_schema_info(self) -> Dict[str, List[str]]:
        """Get schema information from the database."""
        if not self.connection:
            if not self.connect():
                return {}
                
        try:
            cursor = self.connection.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                # Get column info for each table
                cursor.execute("PRAGMA table_info(?)", (table,))
                columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name
                schema_info[table] = columns
                
            return schema_info
            
        except Exception as e:
            print(f"Error getting schema info: {e}")
            return {}
            
    def get_schema_string(self) -> str:
        """Get schema as a formatted string."""
        schema_info = self.get_schema_info()
        
        schema_lines = []
        for table, columns in schema_info.items():
            schema_lines.append(f"Table {table}:")
            for col in columns:
                schema_lines.append(f"  - {col}")
            schema_lines.append("")
            
        return "\n".join(schema_lines)


class ExecutionAccuracyEvaluator:
    """Evaluator for SQL execution accuracy."""
    
    def __init__(self, db_path: str):
        self.sql_executor = SQLExecutor(db_path)
        
    def evaluate_single(
        self,
        predicted_query: str,
        ground_truth_query: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate a single prediction.
        
        Returns:
            Tuple of (is_correct, detailed_results)
        """
        results = {
            'predicted_query': predicted_query,
            'ground_truth_query': ground_truth_query,
            'predicted_success': False,
            'ground_truth_success': False,
            'predicted_result': None,
            'ground_truth_result': None,
            'predicted_error': None,
            'ground_truth_error': None,
            'execution_match': False,
        }
        
        # Execute ground truth query
        gt_success, gt_result, gt_error = self.sql_executor.execute_query(ground_truth_query)
        results['ground_truth_success'] = gt_success
        results['ground_truth_result'] = gt_result
        results['ground_truth_error'] = gt_error
        
        # Execute predicted query
        pred_success, pred_result, pred_error = self.sql_executor.execute_query(predicted_query)
        results['predicted_success'] = pred_success
        results['predicted_result'] = pred_result
        results['predicted_error'] = pred_error
        
        # Compare results
        if gt_success and pred_success:
            results['execution_match'] = self.sql_executor.compare_results(gt_result, pred_result)
        elif not gt_success and not pred_success:
            # Both failed - consider this a match if same error type
            results['execution_match'] = False  # Conservative approach
        else:
            results['execution_match'] = False
            
        return results['execution_match'], results
        
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions.
        
        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
            
        detailed_results = []
        correct_count = 0
        
        print(f"Evaluating {len(predictions)} predictions...")
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            is_correct, details = self.evaluate_single(pred, gt)
            detailed_results.append(details)
            
            if is_correct:
                correct_count += 1
                
            # Progress update
            if (i + 1) % 10 == 0 or i == len(predictions) - 1:
                print(f"Processed {i + 1}/{len(predictions)} samples")
                
        execution_accuracy = correct_count / len(predictions)
        
        # Calculate additional statistics
        pred_success_count = sum(1 for r in detailed_results if r['predicted_success'])
        gt_success_count = sum(1 for r in detailed_results if r['ground_truth_success'])
        
        results = {
            'execution_accuracy': execution_accuracy,
            'correct_predictions': correct_count,
            'total_predictions': len(predictions),
            'predicted_success_rate': pred_success_count / len(predictions),
            'ground_truth_success_rate': gt_success_count / len(predictions),
            'detailed_results': detailed_results,
        }
        
        return results
        
    def evaluate_from_file(
        self,
        predictions_file: str,
        ground_truth_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions from files.
        
        Args:
            predictions_file: JSON file with predicted SQL queries
            ground_truth_file: JSON file with ground truth SQL queries
            output_file: Optional file to save detailed results
        """
        # Load predictions
        with open(predictions_file, 'r') as f:
            pred_data = json.load(f)
            
        # Load ground truths
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
            
        # Extract queries (assuming both files have same structure)
        predictions = []
        ground_truths = []
        
        # Create lookup for ground truth by id
        gt_lookup = {item.get('id', i): item for i, item in enumerate(gt_data)}
        
        for i, pred_item in enumerate(pred_data):
            pred_id = pred_item.get('id', i)
            
            if pred_id in gt_lookup:
                gt_item = gt_lookup[pred_id]
                predictions.append(pred_item.get('predicted_sql', ''))
                ground_truths.append(gt_item.get('query', ''))
            else:
                print(f"Warning: No ground truth found for prediction ID {pred_id}")
                
        # Evaluate
        results = self.evaluate_batch(predictions, ground_truths)
        
        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to {output_file}")
            
        return results
        
    def close(self):
        """Close database connection."""
        self.sql_executor.disconnect()