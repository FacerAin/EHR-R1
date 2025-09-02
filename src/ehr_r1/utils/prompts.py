"""Prompt templates for EHRSQL tasks."""

from typing import Any, Dict
from pathlib import Path


def load_template(template_name: str) -> str:
    """Load template from file.
    
    Args:
        template_name: Template filename (e.g., 'omnisql_prompt.jinja2')
    
    Returns:
        Template content as string
    """
    template_path = Path(__file__).parent.parent / "templates" / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found at {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def render_template(template_content: str, **kwargs) -> str:
    """Render Jinja2 template with variables.
    
    Args:
        template_content: Template content string
        **kwargs: Variables to render in template
    
    Returns:
        Rendered template
    """
    try:
        from jinja2 import Template
        
        template = Template(template_content)
        return template.render(**kwargs)
    except ImportError:
        # Fallback to simple string formatting if Jinja2 not available
        return simple_template_render(template_content, **kwargs)


def simple_template_render(template_content: str, **kwargs) -> str:
    """Simple template rendering without Jinja2.
    
    Args:
        template_content: Template content with {{ variable }} placeholders
        **kwargs: Variables to substitute
    
    Returns:
        Rendered template
    """
    result = template_content
    for key, value in kwargs.items():
        result = result.replace(f"{{{{ {key} }}}}", str(value))
    return result


def format_prompt(template_name: str, **kwargs) -> str:
    """Format prompt using template.
    
    Args:
        template_name: Template filename (e.g., 'omnisql_prompt.jinja2')
        **kwargs: Template variables
    
    Returns:
        Formatted prompt string
    """
    template_content = load_template(template_name)
    return render_template(template_content, **kwargs)


# Convenience functions for common templates
def format_sql_prompt(schema: str, question: str, template: str = "omnisql_prompt.jinja2") -> str:
    """Format SQL generation prompt.
    
    Args:
        schema: Database schema
        question: Natural language question
        template: Template to use
    
    Returns:
        Formatted prompt string
    """
    return format_prompt(template, schema=schema, question=question)


class EHRSQLPromptTemplate:
    """Template manager for EHRSQL prompts."""

    @classmethod
    def create_prompt(
        cls,
        question: str,
        db_details: str,
    ) -> str:
        """Create a formatted prompt using the template."""
        return format_sql_prompt(schema=db_details, question=question)

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
