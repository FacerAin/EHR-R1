"""Prompt templates for EHRSQL tasks."""

from pathlib import Path
from typing import Any, Dict


def load_template(template_name: str) -> str:
    """Load template from file.

    Args:
        template_name: Template filename (e.g., 'omnisql_prompt.jinja2')

    Returns:
        Template content as string
    """
    template_path = Path(__file__).parent.parent / "templates" / template_name

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template {template_name} not found at {template_path}"
        )

    with open(template_path, "r", encoding="utf-8") as f:
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
