"""Project scaffolding for ``llmrivotril init``.

Creates a minimal, ready-to-use project skeleton with an agent module,
guardrails, a starter test, and packaging metadata.
"""

import json
import re
from pathlib import Path

_AGENT_PY = '''\
"""Example RivotrilAgent setup."""

import os

from llmrivotril import RivotrilAgent

from .guardrails import DEFAULT_GUARDRAILS
from .verifier import default_verifier


agent = RivotrilAgent(
    model=os.getenv("RIVOTRIL_MODEL", "gpt-4o-mini"),
    guardrails=DEFAULT_GUARDRAILS,
    verifier=default_verifier,
)


if __name__ == "__main__":
    response = agent.run("Explain what a guardrail is in AI safety.")
    print(response)
'''


_GUARDRAILS_PY = '''\
"""Default guardrails for the project."""

from llmrivotril import Guardrail


DEFAULT_GUARDRAILS = [
    Guardrail(
        name="content-safety",
        allowed_topics=["AI safety", "machine learning"],
        disallowed_keywords=["password", "secret", "token"],
        disallowed_patterns=[r"\\b\\d{3}-\\d{2}-\\d{4}\\b"],
        max_tokens=500,
    )
]
'''


_VERIFIER_PY = '''\
"""Default grounding verifier for the project."""

from llmrivotril.verifier import KeywordOverlapVerifier, Verifier


default_verifier = Verifier(check_fn=KeywordOverlapVerifier(threshold=0.1).as_callable())
'''


_NOTEBOOK_JSON = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# {project_title} Demo\n\n",
                "Interactive walkthrough of the scaffolded agent.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from {project_name}.agent import agent\n",
                "from {project_name}.guardrails import DEFAULT_GUARDRAILS\n",
                "from {project_name}.verifier import default_verifier\n",
                "\n",
                "print('Guardrails:', [g.name for g in DEFAULT_GUARDRAILS])",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Replace with a real API key to run live, or mock the client for tests.\n",
                "response = agent.run('What is a guardrail in AI safety?')\n",
                "print(response)",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


_TEST_AGENT_PY = """\
from unittest.mock import MagicMock, patch

from {project_name}.agent import agent


def test_agent_run_with_mock():
    with patch.object(agent.provider, "complete", return_value=MagicMock(text="Mock response")):
        result = agent.run("hello")
        assert result == "Mock response"
"""


_ENV_EXAMPLE = """\
# Copy to .env and fill in your values
OPENAI_API_KEY=sk-...
RIVOTRIL_MODEL=gpt-4o-mini
RIVOTRIL_REQUEST_TIMEOUT=30
RIVOTRIL_RATE_LIMIT_MAX_CALLS=10
RIVOTRIL_RETRY_MAX_ATTEMPTS=3
"""


_GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env

# IDEs
.vscode/
.idea/

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
"""


_README_MD = """\
# {project_title}

Project scaffolded with `llmrivotril init`.

## Structure

- `{project_name}/agent.py` — pre-configured `RivotrilAgent`
- `{project_name}/guardrails.py` — default guardrails
- `{project_name}/verifier.py` — default grounding verifier
- `notebooks/demo.ipynb` — interactive demo
- `tests/test_agent.py` — starter test

## Run

```bash
pip install -e .
python {project_name}/agent.py
```

## Test

```bash
pytest
```
"""


_PYPROJECT_TOML = """\
[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[project]
name = "{project_name}"
version = "0.1.0"
description = "LLM-Rivotril powered project"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "llmrivotril>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
"""


def _sanitize_name(name: str) -> str:
    """Turn any directory name into a valid Python package name."""
    sanitized = re.sub(r"[^a-z0-9_]", "_", name.lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "project_" + (sanitized or "llmrivotril")
    return sanitized


def create_project(path: str | Path, name: str | None = None) -> Path:
    """Create the project skeleton at ``path``.

    Existing files are skipped so the command is safe to re-run.
    """
    root = Path(path).expanduser().resolve()
    project_name = _sanitize_name(name or root.name)

    src_dir = root / project_name
    tests_dir = root / "tests"
    notebooks_dir = root / "notebooks"

    directories = [root, src_dir, tests_dir, notebooks_dir]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    files = {
        src_dir / "__init__.py": "",
        src_dir / "agent.py": _AGENT_PY,
        src_dir / "guardrails.py": _GUARDRAILS_PY,
        src_dir / "verifier.py": _VERIFIER_PY,
        tests_dir / "__init__.py": "",
        tests_dir / "test_agent.py": _TEST_AGENT_PY.format(project_name=project_name),
        root / ".env.example": _ENV_EXAMPLE,
        root / ".gitignore": _GITIGNORE,
        root / "README.md": _README_MD.format(
            project_title=project_name.replace("_", " ").title(),
            project_name=project_name,
        ),
        root / "pyproject.toml": _PYPROJECT_TOML.format(project_name=project_name),
        notebooks_dir / "demo.ipynb": json.dumps(
            _NOTEBOOK_JSON,
            indent=2,
        )
        .replace("{project_title}", project_name.replace("_", " ").title())
        .replace("{project_name}", project_name),
    }

    for file_path, content in files.items():
        if file_path.exists():
            continue
        file_path.write_text(content, encoding="utf-8")

    return root
