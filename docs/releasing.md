# Releasing

[← Back to README](../README.md) · *[Português](releasing.pt-BR.md)*

## Pre-release checklist

1. Confirm the version in `pyproject.toml`, `src/llmrivotril/__init__.py`, and
   `CHANGELOG.md` is identical.
   The README version badge is sourced dynamically from PyPI, so it must not be
   edited manually. The test suite also checks that the package version matches
   `pyproject.toml`.
2. Confirm the package author and PyPI project name are final.
3. Install CI/development dependencies through
   `.github/constraints-ci.txt`, as the workflows do.
4. Run the lint, format, type-check, and test gates.
5. Run `python -m build` and `python -m twine check dist/*`.
6. Install the wheel in a clean virtual environment and run the CLI smoke test.
7. Run the provider and vector-store checks listed in the technical review.
8. Upload to TestPyPI first when validating a new release process.

## PyPI release

The repository contains `.github/workflows/pypi-publish.yml`, which builds the
wheel and source distribution and publishes them with the legacy PyPI API
token stored as the `PYPI_API_TOKEN` repository secret. Trusted Publishing is
temporarily paused because GitHub Actions is returning an unexplained
zero-job `startup_failure` for the OIDC workflow.

One-time setup:

1. In the PyPI project settings, open **Publishing** and add a GitHub Actions
   trusted publisher with:
   - owner: `ailake-io`
   - repository: `llmrivotril`
   - secret: `PYPI_API_TOKEN` configured in repository Actions secrets
2. In the repository's `pypi` environment, keep the five-minute wait timer.

After setup, publishing a GitHub Release from a version tag triggers the
workflow automatically. A rerun must use a new version if any files were
already uploaded, because PyPI release files are immutable.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Current release status

Version `0.1.6` contains the persistence/logging security hardening. Merge the
release PR into `main`, wait for the post-merge CI, then create and publish the
`v0.1.6` GitHub Release to trigger the PyPI workflow. Verify the package on
PyPI.

The CI toolchain is pinned in `.github/constraints-ci.txt`. This constrains
linting, type checking, testing, and packaging tools without forcing exact
runtime/provider versions on applications that install `llmrivotril`.
