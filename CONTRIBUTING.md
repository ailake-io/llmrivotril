# Contributing to LLM-Rivotril

Thank you for your interest in improving LLM-Rivotril!

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/llmrivotril.git
   cd llmrivotril
   ```
3. Create a virtual environment and install in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, adding tests for new behavior.
3. Run the test suite:
   ```bash
   pytest
   ```
4. Run linting and type checks:
   ```bash
   ruff check src tests
   ruff format --check src tests
   mypy src
   ```
5. Commit your changes and open a pull request.

## Code Style

- Follow PEP 8.
- Use type hints for public functions and classes.
- Keep functions focused and small.
- Write docstrings for modules, classes, and public methods.

## Testing

- Place unit tests in `tests/`.
- Use `pytest` fixtures to isolate stateful components such as `MemoryStore` and `MetricsCollector`.
- Mock external API calls (OpenAI / Instructor) in agent tests.

## Reporting Issues

When reporting bugs, include:

- Python version
- Package version (`pip show llmrivotril`)
- Minimal reproducible example
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
