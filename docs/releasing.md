# Releasing

[← Back to README](../README.md)

## Pre-release checklist

1. Confirm the version in `pyproject.toml`, `src/llmrivotril/__init__.py`, and
   `CHANGELOG.md` is identical.
2. Confirm the package author and PyPI project name are final.
3. Run the lint, format, type-check, and test gates.
4. Run `python -m build` and `python -m twine check dist/*`.
5. Install the wheel in a clean virtual environment and run the CLI smoke test.
6. Run the provider and vector-store checks listed in the technical review.
7. Upload to TestPyPI first when validating a new release process.

## PyPI release

The repository contains `.github/workflows/release.yml`. Configure its `pypi`
GitHub environment as a PyPI Trusted Publisher for the exact repository and
workflow, then publish a GitHub Release from the matching tag. The workflow
builds both the wheel and source distribution and publishes them without a
long-lived API token.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Current release status

As of 29/09/2026, version `0.1.0` has passed the local quality gates and the
clean-environment import/CLI smoke test. The remaining work is release
operations and live-service validation, not a known critical code fix. See the
[technical review](technical-review.md#status-validado-em-29092026) for the
complete checklist and known integration limitations.
