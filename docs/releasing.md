# Releasing

[← Back to README](../README.md) · *[Português](releasing.pt-BR.md)*

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

The repository contains `.github/workflows/release.yml`, which builds the
wheel and source distribution and publishes them with `twine` using a PyPI
API token. One-time setup:

1. Generate an API token on PyPI (account-wide for the first upload; scope it
   to the `llmrivotril` project once it exists there).
2. Add it as a repository secret named `PYPI_API_TOKEN`
   (Settings → Secrets and variables → Actions).

After that, publishing a GitHub Release from a version tag triggers the
workflow automatically. `twine upload --skip-existing` makes a rerun of the
workflow (e.g. after a transient failure) safe -- it won't error on files
already uploaded for that version.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Current release status

As of 29/09/2026, version `0.1.0` has passed the local quality gates and the
clean-environment import/CLI smoke test. The remaining work is release
operations and live-service validation, not a known critical code fix. See the
[technical review](technical-review.md#status-validado-em-29092026) for the
complete checklist and known integration limitations.
