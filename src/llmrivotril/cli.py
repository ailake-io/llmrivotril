import os

import click

from . import __version__
from .scaffold import create_project
from .server import run_dashboard


@click.group()
@click.version_option(version=__version__, prog_name="llmrivotril")
def cli() -> None:
    """LLM-Rivotril CLI tools for agent safety and metrics."""


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host for the local web dashboard.")
@click.option("--port", default=8000, help="Port for the local web dashboard.")
def dashboard(host: str, port: int) -> None:
    """Launch the local web dashboard to monitor token consumption and guardrails."""
    click.echo(f"Starting LLM-Rivotril dashboard at http://{host}:{port} ...")
    run_dashboard(host=host, port=port)


@cli.command()
@click.option("--mock", is_flag=True, default=True, help="Use deterministic mock responses.")
@click.option("--output", default=None, help="Path to write the JSON report.")
def benchmark(mock: bool, output: str | None) -> None:
    """Run the built-in red-team benchmark (mock mode, no API cost)."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "scripts.benchmark"]
    if mock:
        cmd.append("--mock")
    if output:
        cmd.extend(["--output", output])
    subprocess.run(cmd, check=False)


@cli.command()
def doctor() -> None:
    """Check the local environment for common configuration issues."""
    issues = 0

    if not os.environ.get("OPENAI_API_KEY"):
        click.echo("⚠️  OPENAI_API_KEY is not set (only required for live API mode).")
        issues += 1
    else:
        click.echo("✓ OPENAI_API_KEY is set.")

    if os.environ.get("RIVOTRIL_DASHBOARD_TOKEN"):
        click.echo("✓ RIVOTRIL_DASHBOARD_TOKEN is set (dashboard will require Bearer token).")
    else:
        click.echo("ℹ️  RIVOTRIL_DASHBOARD_TOKEN is not set (dashboard is open).")

    try:
        import sentence_transformers  # noqa: F401

        click.echo("✓ sentence-transformers is installed (semantic extras available).")
    except ImportError:
        click.echo(
            "ℹ️  sentence-transformers is not installed; "
            "install with: pip install llmrivotril[semantic]"
        )

    provider_packages = {
        "anthropic": "anthropic",
        "cohere": "cohere",
        "gemini": "google.generativeai",
    }
    installed = []
    missing = []
    for name, module in provider_packages.items():
        try:
            __import__(module)
            installed.append(name)
        except ImportError:
            missing.append(name)

    if installed:
        click.echo(f"✓ Provider SDKs installed: {', '.join(installed)}.")
    if missing:
        click.echo(
            "ℹ️  Optional provider SDKs not installed: "
            f"{', '.join(missing)}. Install with: pip install llmrivotril[providers]"
        )

    if issues == 0:
        click.echo("\nEnvironment looks good for mock/live usage.")
    else:
        click.echo(f"\nFound {issues} issue(s).")


@cli.command()
def version() -> None:
    """Show the installed LLM-Rivotril version."""
    click.echo(__version__)


@cli.command()
@click.argument("path", default=".", required=False)
@click.option("--name", default=None, help="Project/package name (defaults to directory name).")
def init(path: str, name: str | None) -> None:
    """Initialize a new LLM-Rivotril project skeleton."""
    root = create_project(path, name)
    click.echo(f"Initialized LLM-Rivotril project at {root}")
    click.echo("Run:")
    click.echo(f"  cd {root}")
    click.echo("  pip install -e .")
    click.echo("  pytest")


if __name__ == "__main__":
    cli()
