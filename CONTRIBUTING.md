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
   For a lighter install that skips optional provider SDKs and semantic dependencies,
   use `pip install -e ".[ci]"`.

## Development Workflow

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop && git pull
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, adding tests for new behavior.
3. Run the test suite:
   ```bash
   pytest
   ```
4. Run linting and type checks:
   ```bash
   ruff check src tests examples scripts
   ruff format --check src tests examples scripts
   mypy src
   ```
5. Commit your changes and open a pull request against `develop`.

## Branching and Merging

- `main` holds released code; `develop` is the integration branch.
- Feature and fix PRs target `develop` and are **squash-merged**.
- Group related changes in one PR instead of a chain of follow-ups.
- Releases go `develop` → `main` through a PR merged with a **merge commit**
  (never squash). Squashing creates a commit `develop` does not have, so the
  branches diverge and the next release PR conflicts.
- Hotfixes merged straight into `main` must be merged back into `develop`
  right away.

## CI and Branch Protection

`main` requires these check names: `lint-and-test (3.10)` to `(3.13)`,
`build`, `extras`, `Dependency audit` and `Analyze Python` (CodeQL).

- Renaming a job or turning it into a matrix changes its check name. Keep the
  required name (for example with an aggregate job, as `extras` does) or update
  branch protection in the same change.
- Workflows only run actions in the repository allowlist, which is pinned by
  commit SHA. When Dependabot bumps an action, a maintainer must add the new SHA
  under Settings → Actions → General before merging; otherwise the workflow
  ends in `startup_failure`, which `gh pr checks` does not report.

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
