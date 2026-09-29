# Releasing

[← Back to README](../README.md)

## Pre-release checklist

1. Confirm the version in `pyproject.toml`, `src/llmrivotril/__init__.py`, and
   `CHANGELOG.md` is identical.
2. Run the lint, format, type-check, and test gates.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Install the wheel in a clean virtual environment and run the CLI smoke test.
5. Upload to TestPyPI first when validating a new release process.

## PyPI release

The repository contains `.github/workflows/release.yml`. Configure its `pypi`
GitHub environment as a PyPI Trusted Publisher for the exact repository and
workflow, then publish a GitHub Release from the matching tag. The workflow
builds both the wheel and source distribution and publishes them without a
long-lived API token.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.
