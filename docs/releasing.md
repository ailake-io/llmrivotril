# Releasing

[← Back to README](../README.md) · *[Português](releasing.pt-BR.md)*

## Pre-release checklist

1. Confirm the version in `pyproject.toml`, `src/llmrivotril/__init__.py`, and
   `CHANGELOG.md` is identical.
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
wheel and source distribution and publishes them with `twine` using a PyPI
API token. One-time setup:

1. Generate an API token on PyPI. For an existing project, scope it to
   `llmrivotril` whenever PyPI allows that.
2. Add it as a repository secret named `PYPI_API_TOKEN`
   (Settings → Secrets and variables → Actions).
3. Keep the token project-scoped. If an account-wide token was used for the
   initial upload, revoke it and replace the secret with a token scoped only
   to `llmrivotril`.
4. In the repository's `pypi` GitHub Environment (Settings → Environments),
   consider adding required reviewers/a wait timer. `release.yml` references
   this environment specifically so its protection rules gate the actual
   PyPI upload step -- this is the mitigation for using a static long-lived
   token instead of OIDC Trusted Publishing, which has no equivalent
   standing credential to protect.

After setup, publishing a GitHub Release from a version tag triggers the
workflow automatically. `twine upload --skip-existing` makes a rerun of the
workflow (e.g. after a transient failure) safe -- it won't error on files
already uploaded for that version.

Do not reuse a version that has already been uploaded: PyPI release files are
immutable. Update the changelog and package version before creating a new tag.

## Current release status

As of 30/09/2026, version `0.1.4` is ready for release: the correction is
merged into `main` and the CI gates have passed. Create and publish the
`v0.1.4` GitHub Release to trigger the PyPI workflow, then verify the package
on PyPI. See the [technical review](technical-review.md#status-validado-em-29092026)
for the complete checklist and known integration limitations.

The CI toolchain is pinned in `.github/constraints-ci.txt`. This constrains
linting, type checking, testing, and packaging tools without forcing exact
runtime/provider versions on applications that install `llmrivotril`.
