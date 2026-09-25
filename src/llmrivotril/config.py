"""Environment-variable and file-based configuration helpers.

Variables are read lazily when ``RivotrilAgent`` is instantiated. Explicit
constructor arguments always override environment values, which in turn override
values read from configuration files.

Supported configuration sources, in ascending order of precedence:

1. ``.env`` file in the current working directory
2. ``pyproject.toml`` section ``[tool.llmrivotril]``
3. ``llmrivotril.toml`` in the current working directory
4. File pointed to by ``RIVOTRIL_CONFIG_FILE``
5. Environment variables prefixed with ``RIVOTRIL_``
6. Explicit constructor arguments to ``RivotrilAgent``
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("llmrivotril")

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

_ENV_FLOATS = {
    "RIVOTRIL_REQUEST_TIMEOUT",
    "RIVOTRIL_RATE_LIMIT_MAX_CALLS",
    "RIVOTRIL_RATE_LIMIT_PER_SECONDS",
    "RIVOTRIL_RETRY_MIN_WAIT",
    "RIVOTRIL_RETRY_MAX_WAIT",
    "RIVOTRIL_CIRCUIT_RECOVERY_TIMEOUT",
}

_ENV_INTS = {
    "RIVOTRIL_RETRY_MAX_ATTEMPTS",
    "RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD",
    "RIVOTRIL_MAX_SESSION_TOKENS",
    "RIVOTRIL_MAX_PROMPT_TOKENS",
}

_ENV_STRINGS = {
    "RIVOTRIL_MODEL",
    "RIVOTRIL_API_KEY",
    "RIVOTRIL_BASE_URL",
    "RIVOTRIL_SYSTEM_PROMPT",
    "RIVOTRIL_PROVIDER",
    "RIVOTRIL_METRICS_PATH",
}

_ENV_BOOLS = {
    "RIVOTRIL_ENABLE_LAZY_CLIENTS",
}

_ALL_KNOWN_KEYS = _ENV_STRINGS | _ENV_FLOATS | _ENV_INTS | _ENV_BOOLS


def _to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _to_kwarg(env_key: str) -> str:
    """Convert ``RIVOTRIL_RATE_LIMIT_MAX_CALLS`` to ``rate_limit_max_calls``."""
    return env_key.replace("RIVOTRIL_", "").lower()


def _coerce_file_value(key: str, value: object) -> object:
    """Coerce a value read from a config file to the expected Python type.

    File values are assumed to already have the right type when TOML is used.
    This helper only normalizes booleans from strings when YAML returns plain
    strings.
    """
    if key.upper() in _ENV_BOOLS and isinstance(value, str):
        return _to_bool(value)
    return value


def _read_toml(path: Path) -> dict[str, object]:
    """Parse a TOML file and return the ``[tool.llmrivotril]`` or root table."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    # For pyproject.toml we look under [tool.llmrivotril]; for a dedicated file
    # we accept both the namespaced section and the root table, with the
    # namespaced section taking precedence.
    if "tool" in data and isinstance(data["tool"], dict):
        tool_llmrivotril = data["tool"].get("llmrivotril")
        if isinstance(tool_llmrivotril, dict):
            return {str(k): v for k, v in tool_llmrivotril.items()}

    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if not isinstance(v, dict)}

    return {}


def _read_yaml(path: Path) -> dict[str, object]:
    """Parse a YAML config file if PyYAML is available."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configuration files. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if not isinstance(v, dict)}


def _read_config_file(path: Path) -> dict[str, object]:
    """Read a single config file, detecting TOML or YAML by extension."""
    suffix = path.suffix.lower()
    if suffix in {".toml"}:
        raw = _read_toml(path)
    elif suffix in {".yaml", ".yml"}:
        raw = _read_yaml(path)
    else:
        # Default to TOML for extensionless or unknown files to match the
        # original spec intent.
        raw = _read_toml(path)

    return {str(k): _coerce_file_value(str(k), v) for k, v in raw.items()}


def _find_config_file() -> Path | None:
    """Return the first existing config file in the search order."""
    env_path = os.environ.get("RIVOTRIL_CONFIG_FILE")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        return None

    candidates = [
        Path.cwd() / "llmrivotril.toml",
        Path.cwd() / "llmrivotril.yaml",
        Path.cwd() / "llmrivotril.yml",
        Path.cwd() / "pyproject.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_file_config(path: str | Path | None = None) -> dict[str, object]:
    """Return a dict of RivotrilAgent-compatible kwargs from a config file.

    If ``path`` is not provided, search order is:
    ``RIVOTRIL_CONFIG_FILE`` > ``llmrivotril.toml`` > ``llmrivotril.yaml`` >
    ``pyproject.toml``.
    """
    if path is not None:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return {}
        return _read_config_file(file_path)

    found_path = _find_config_file()
    if found_path is None:
        return {}

    config = _read_config_file(found_path)

    # When falling back to pyproject.toml, only keep keys the agent understands.
    if found_path.name == "pyproject.toml":
        config = {k: v for k, v in config.items() if _to_env_key(k) in _ALL_KNOWN_KEYS}

    return config


def load_env_config() -> dict[str, object]:
    """Return a dict of RivotrilAgent-compatible kwargs from environment variables."""
    config: dict[str, object] = {}

    for key in _ENV_STRINGS:
        value = os.environ.get(key)
        if value is not None:
            config[_to_kwarg(key)] = value

    for key in _ENV_FLOATS:
        value = os.environ.get(key)
        if value is not None:
            config[_to_kwarg(key)] = float(value)

    for key in _ENV_INTS:
        value = os.environ.get(key)
        if value is not None:
            config[_to_kwarg(key)] = int(value)

    for key in _ENV_BOOLS:
        value = os.environ.get(key)
        if value is not None:
            config[_to_kwarg(key)] = _to_bool(value)

    return config


def _load_dotenv() -> None:
    """Load a local ``.env`` file if ``python-dotenv`` is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def load_config() -> dict[str, object]:
    """Merge file and environment configuration.

    Precedence: file < environment.
    """
    _load_dotenv()
    file_config = load_file_config()
    if "api_key" in file_config:
        logger.warning(
            "Reading API key from %s. Consider using environment variables or a secret manager.",
            _find_config_file(),
        )
    env_config = load_env_config()
    merged: dict[str, object] = dict(file_config)
    merged.update(env_config)
    return merged


def _to_env_key(kwarg_key: str) -> str:
    """Convert ``rate_limit_max_calls`` back to ``RIVOTRIL_RATE_LIMIT_MAX_CALLS``."""
    return f"RIVOTRIL_{kwarg_key.upper()}"
