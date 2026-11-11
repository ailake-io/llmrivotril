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
temporarily paused: the repository Actions allowlist blocked the artifact
actions and the PyPA publish action, causing a zero-job `startup_failure`.
This was a policy failure, not a PyPI authentication failure.

One-time setup:

1. Create a PyPI API token scoped to `llmrivotril` and store it in the
   `ailake-io/llmrivotril` repository Actions secret `PYPI_API_TOKEN`.
2. In the repository's `pypi` environment, keep the five-minute wait timer.

After setup, publishing a GitHub Release from a version tag triggers the
workflow automatically. To recover an older tag with the current workflow, run
`gh workflow run pypi-publish.yml --ref main -f tag=vX.Y.Z`.
Release events use the workflow from the tagged commit; fixing `main` does
not repair a workflow embedded in an existing tag. PyPI files are immutable;
use a new version for changed distributions.

Keep the API token until a successful OIDC publication has been verified.
To restore OIDC, allow the exact pinned publish action in the repository or
organization Actions policy, configure the matching workflow and `pypi`
environment in PyPI, and verify a publication before revoking the token and
deleting the GitHub secret.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Previous release record (0.1.6)

Version `0.1.6` was published successfully with a wheel and source distribution
using the token workflow. Changes under `[Unreleased]` are not included in
that published tag, even when they have subsequently been merged into `main`.

The CI toolchain is pinned in `.github/constraints-ci.txt`. This constrains
linting, type checking, testing, and packaging tools without forcing exact
runtime/provider versions on applications that install `llmrivotril`.
