"""Prompt templates for EHRSQL tasks."""

from typing import Any, Dict


class EHRSQLPromptTemplate:
    """Template manager for EHRSQL prompts."""

    INPUT_PROMPT_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""

    @classmethod
    def create_prompt(
        cls,
        question: str,
        db_details: str,
    ) -> str:
        """Create a formatted prompt using the template."""
        return cls.INPUT_PROMPT_TEMPLATE.format(
            question=question,
            db_details=db_details,
        )

    @classmethod
    def extract_sql_from_response(cls, response: str) -> str:
        """Extract SQL query from model response."""
        # Look for SQL in code blocks
        lines = response.split("\n")
        in_code_block = False
        sql_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    break  # End of code block
                else:
                    in_code_block = True  # Start of code block
                    continue

            if in_code_block:
                # Skip comment lines that start with --
                if not line.strip().startswith("--"):
                    sql_lines.append(line)

        if sql_lines:
            sql_query = "\n".join(sql_lines).strip()
        else:
            # Fallback: try to extract SQL without code blocks
            sql_query = cls._extract_sql_fallback(response)

        return sql_query

    @classmethod
    def _extract_sql_fallback(cls, response: str) -> str:
        """Fallback method to extract SQL when code blocks are not found."""
        # Look for common SQL keywords to identify SQL query
        sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
        lines = response.split("\n")

        sql_lines = []
        found_sql = False

        for line in lines:
            line_upper = line.strip().upper()

            # Check if this line starts with SQL keyword
            if any(line_upper.startswith(keyword) for keyword in sql_keywords):
                found_sql = True

            if found_sql:
                # Stop at empty line or explanation text
                if not line.strip() or line.strip().lower().startswith(
                    ("this query", "the query", "explanation")
                ):
                    break
                sql_lines.append(line)

        return "\n".join(sql_lines).strip() if sql_lines else response.strip()


# For backward compatibility and convenience
def create_ehrsql_prompt(question: str, db_details: str) -> str:
    """Create EHRSQL prompt - convenience function."""
    return EHRSQLPromptTemplate.create_prompt(question, db_details)


def extract_sql_from_response(response: str) -> str:
    """Extract SQL from model response - convenience function."""
    return EHRSQLPromptTemplate.extract_sql_from_response(response)
