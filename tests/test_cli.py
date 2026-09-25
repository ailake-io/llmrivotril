from unittest.mock import patch

from click.testing import CliRunner

from llmrivotril.cli import cli


def test_init_command(tmp_path):
    runner = CliRunner()
    target = tmp_path / "my_bot"
    result = runner.invoke(cli, ["init", str(target)])
    assert result.exit_code == 0
    assert (target / "pyproject.toml").exists()
    assert (target / "my_bot" / "agent.py").exists()
    assert (target / "my_bot" / "guardrails.py").exists()
    assert (target / "my_bot" / "verifier.py").exists()
    assert (target / "tests" / "test_agent.py").exists()
    assert (target / ".env.example").exists()
    assert (target / "notebooks" / "demo.ipynb").exists()


def test_init_command_sanitizes_name(tmp_path):
    runner = CliRunner()
    target = tmp_path / "99-Bot!"
    result = runner.invoke(cli, ["init", str(target)])
    assert result.exit_code == 0
    # Names starting with a digit are prefixed with "project_".
    assert (target / "project_99_bot" / "agent.py").exists()


def test_dashboard_command():
    runner = CliRunner()
    with patch("llmrivotril.cli.run_dashboard") as mock_run:
        result = runner.invoke(cli, ["dashboard", "--port", "9999"])
    assert result.exit_code == 0
    assert "Starting LLM-Rivotril dashboard" in result.output
    mock_run.assert_called_once_with(host="127.0.0.1", port=9999)


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_doctor_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.output


def test_benchmark_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "mock" in result.output.lower()
