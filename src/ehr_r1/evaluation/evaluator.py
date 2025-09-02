"""Evaluation metrics and evaluator for EHRSQL."""

from typing import Dict, List, Tuple, Any, Optional
import sqlparse
from sklearn.metrics import accuracy_score
import pandas as pd
import json
from ..utils.sql_executor import ExecutionAccuracyEvaluator, SQLExecutor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EHRSQLEvaluator:
    """Evaluator for EHRSQL model performance."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.execution_evaluator = None
        if db_path:
            self.execution_evaluator = ExecutionAccuracyEvaluator(db_path)
        
    def compute_execution_accuracy(
        self,
        predicted_queries: List[str],
        ground_truth_queries: List[str],
    ) -> Dict[str, Any]:
        """Compute execution accuracy."""
        if not self.execution_evaluator:
            raise ValueError("Database path not provided. Cannot compute execution accuracy.")
            
        return self.execution_evaluator.evaluate_batch(predicted_queries, ground_truth_queries)
        
    def compute_exact_match(
        self,
        predicted_queries: List[str],
        ground_truth_queries: List[str],
    ) -> float:
        """Compute exact match accuracy."""
        if len(predicted_queries) != len(ground_truth_queries):
            raise ValueError("Number of predicted and ground truth queries must match")
            
        exact_matches = 0
        for pred, gt in zip(predicted_queries, ground_truth_queries):
            # Normalize queries for comparison
            pred_normalized = self._normalize_query(pred)
            gt_normalized = self._normalize_query(gt)
            
            if pred_normalized == gt_normalized:
                exact_matches += 1
                
        return exact_matches / len(predicted_queries)
        
    def _normalize_query(self, query: str) -> str:
        """Normalize SQL query for comparison."""
        if not query:
            return ""
            
        # Remove extra whitespace and convert to lowercase
        query = query.strip().lower()
        
        # Parse and format SQL for consistent comparison
        try:
            parsed = sqlparse.parse(query)[0]
            return sqlparse.format(str(parsed), strip_comments=True, reindent=True)
        except:
            # If parsing fails, return cleaned version
            return ' '.join(query.split())
            
    def parse_sql_components(self, sql_query: str) -> Dict[str, Any]:
        """Parse SQL query components."""
        components = {
            'select_clause': [],
            'from_clause': [],
            'where_clause': [],
            'group_by_clause': [],
            'having_clause': [],
            'order_by_clause': [],
            'keywords': [],
            'tables': [],
            'columns': [],
        }
        
        try:
            parsed = sqlparse.parse(sql_query)[0]
            
            for token in parsed.flatten():
                if token.ttype is sqlparse.tokens.Keyword:
                    components['keywords'].append(token.value.upper())
                elif token.ttype is sqlparse.tokens.Name:
                    # Could be table or column name
                    components['columns'].append(token.value)
                    
        except Exception as e:
            logger.debug(f"Error parsing SQL: {e}")
            
        return components
        
    def compute_component_accuracy(
        self,
        predicted_queries: List[str],
        ground_truth_queries: List[str],
    ) -> Dict[str, float]:
        """Compute accuracy for different SQL components."""
        if len(predicted_queries) != len(ground_truth_queries):
            raise ValueError("Number of predicted and ground truth queries must match")
            
        component_matches = {
            'select_match': 0,
            'from_match': 0,
            'where_match': 0,
            'keyword_match': 0,
        }
        
        for pred, gt in zip(predicted_queries, ground_truth_queries):
            pred_components = self.parse_sql_components(pred)
            gt_components = self.parse_sql_components(gt)
            
            # Compare keywords
            pred_keywords = set(pred_components['keywords'])
            gt_keywords = set(gt_components['keywords'])
            if pred_keywords == gt_keywords:
                component_matches['keyword_match'] += 1
                
        total = len(predicted_queries)
        return {k: v / total for k, v in component_matches.items()}
        
    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation."""
        results = {}
        
        # Exact match accuracy
        results['exact_match_accuracy'] = self.compute_exact_match(predictions, ground_truths)
        
        # Component accuracy
        results.update(self.compute_component_accuracy(predictions, ground_truths))
        
        # Execution accuracy (if database available)
        if self.execution_evaluator:
            exec_results = self.compute_execution_accuracy(predictions, ground_truths)
            results['execution_accuracy'] = exec_results['execution_accuracy']
            results['predicted_success_rate'] = exec_results['predicted_success_rate']
            results['ground_truth_success_rate'] = exec_results['ground_truth_success_rate']
            
            # Save detailed execution results if requested
            if output_file:
                detailed_file = output_file.replace('.json', '_detailed.json')
                with open(detailed_file, 'w') as f:
                    json.dump(exec_results, f, indent=2)
                logger.info(f"Detailed execution results saved to {detailed_file}")
        
        # Summary statistics
        results['total_predictions'] = len(predictions)
        results['non_empty_predictions'] = sum(1 for p in predictions if p.strip())
        results['empty_prediction_rate'] = 1 - (results['non_empty_predictions'] / len(predictions))
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
            
        return results
        
    def close(self):
        """Close database connections."""
        if self.execution_evaluator:
            self.execution_evaluator.close()