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

The repository contains `.github/workflows/release.yml`, which builds the
wheel and source distribution in an isolated job and publishes them using
PyPI Trusted Publishing (OIDC). No PyPI API token is stored in GitHub.

One-time setup:

1. In the PyPI project settings, open **Publishing** and add a GitHub Actions
   trusted publisher with:
   - owner: `ailake-io`
   - repository: `llmrivotril`
   - workflow: `.github/workflows/release.yml`
   - environment: `pypi`
2. In the repository's `pypi` environment, keep the five-minute wait timer and
   restrict deployment tags to the release tags used by the project. Required
   reviewers are optional for this single-maintainer project.
3. After a successful OIDC publication, revoke the legacy `PYPI_API_TOKEN`
   secret/token used by the previous workflow.

After setup, publishing a GitHub Release from a version tag triggers the
workflow automatically. The official PyPA publish action also creates PyPI
publish attestations by default. A rerun must use a new version if any files
were already uploaded, because PyPI release files are immutable.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Current release status

Version `0.1.5` contains the streaming/tool-execution performance improvements,
the Ubuntu 26.04 CI migration, and the updated Ollama/local-model
documentation. Merge the release PR into `main`, wait for the post-merge CI,
then create and publish the `v0.1.5` GitHub Release to trigger the PyPI
workflow. Verify the package on PyPI and run the Ollama smoke test locally.

The CI toolchain is pinned in `.github/constraints-ci.txt`. This constrains
linting, type checking, testing, and packaging tools without forcing exact
runtime/provider versions on applications that install `llmrivotril`.
