import os

import pytest

from llmrivotril.config import load_config, load_env_config, load_file_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove RIVOTRIL_* variables before each test."""
    for key in list(os.environ):
        if key.startswith("RIVOTRIL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Provide a temporary directory and monkeypatch cwd to it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_load_env_config_returns_empty_dict_when_no_vars():
    assert load_env_config() == {}


def test_load_env_config_reads_strings():
    os.environ["RIVOTRIL_MODEL"] = "gpt-4o"
    os.environ["RIVOTRIL_API_KEY"] = "sk-test"
    os.environ["RIVOTRIL_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["RIVOTRIL_SYSTEM_PROMPT"] = "You are a test assistant."

    config = load_env_config()
    assert config["model"] == "gpt-4o"
    assert config["api_key"] == "sk-test"
    assert config["base_url"] == "http://localhost:11434/v1"
    assert config["system_prompt"] == "You are a test assistant."


def test_load_env_config_reads_floats():
    os.environ["RIVOTRIL_REQUEST_TIMEOUT"] = "15.5"
    os.environ["RIVOTRIL_RATE_LIMIT_PER_SECONDS"] = "2.5"

    config = load_env_config()
    assert config["request_timeout"] == 15.5
    assert config["rate_limit_per_seconds"] == 2.5


def test_load_env_config_reads_ints():
    os.environ["RIVOTRIL_RETRY_MAX_ATTEMPTS"] = "5"
    os.environ["RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD"] = "10"

    config = load_env_config()
    assert config["retry_max_attempts"] == 5
    assert config["circuit_failure_threshold"] == 10


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_load_env_config_reads_bools(value, expected):
    os.environ["RIVOTRIL_ENABLE_LAZY_CLIENTS"] = value
    config = load_env_config()
    assert config["enable_lazy_clients"] is expected


def test_load_file_config_returns_empty_dict_when_no_file(temp_config):
    assert load_file_config() == {}


def test_load_file_config_reads_llmrivotril_toml(temp_config):
    config_path = temp_config / "llmrivotril.toml"
    config_path.write_text(
        'model = "gpt-4o-mini"\n'
        'api_key = "file-key"\n'
        "request_timeout = 25.0\n"
        "retry_max_attempts = 7\n"
        "enable_lazy_clients = true\n"
    )

    config = load_file_config()
    assert config["model"] == "gpt-4o-mini"
    assert config["api_key"] == "file-key"
    assert config["request_timeout"] == 25.0
    assert config["retry_max_attempts"] == 7
    assert config["enable_lazy_clients"] is True


def test_load_file_config_reads_pyproject_toml(temp_config):
    pyproject = temp_config / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "example"\n\n'
        "[tool.llmrivotril]\n"
        'model = "gpt-4o"\n'
        'system_prompt = "from file"\n'
        "request_timeout = 30.0\n"
    )

    config = load_file_config()
    assert config["model"] == "gpt-4o"
    assert config["system_prompt"] == "from file"
    assert config["request_timeout"] == 30.0


def test_load_file_config_ignores_unrelated_pyproject_keys(temp_config):
    pyproject = temp_config / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "example"\n\n'
        "[tool.llmrivotril]\n"
        'model = "gpt-4o"\n'
        'unknown_key = "ignored"\n'
    )

    config = load_file_config()
    assert config["model"] == "gpt-4o"
    assert "unknown_key" not in config


def test_load_file_config_explicit_path(temp_config):
    config_path = temp_config / "custom" / "settings.toml"
    config_path.parent.mkdir()
    config_path.write_text('model = "custom-model"\n')

    config = load_file_config(config_path)
    assert config["model"] == "custom-model"


def test_load_file_config_respects_env_override(temp_config):
    config_path = temp_config / "llmrivotril.toml"
    config_path.write_text('model = "file-model"\n')
    os.environ["RIVOTRIL_MODEL"] = "env-model"

    config = load_config()
    assert config["model"] == "env-model"


def test_load_config_precedence_env_over_file(temp_config):
    config_path = temp_config / "llmrivotril.toml"
    config_path.write_text('model = "file-model"\nrequest_timeout = 10.0\n')
    os.environ["RIVOTRIL_MODEL"] = "env-model"

    config = load_config()
    assert config["model"] == "env-model"
    assert config["request_timeout"] == 10.0
