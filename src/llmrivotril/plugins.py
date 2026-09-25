"""Plugin discovery and loading for llmrivotril extensions.

Plugins are registered via Python package entry points in one of these groups:

- ``llmrivotril.guardrails`` -> must return a ``Guardrail`` instance or class.
- ``llmrivotril.verifiers`` -> must return a grounding verifier callable.

A discovered entry point can be loaded by passing its advertised name to the
agent constructor, either as a string or as part of a list. Use
``plugins="auto"`` to load every discovered plugin automatically.
"""

import logging
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger("llmrivotril")

_GUARD_ENTRY_POINT_GROUP = "llmrivotril.guardrails"
_VERIFIER_ENTRY_POINT_GROUP = "llmrivotril.verifiers"


def discover_plugins() -> dict[str, dict[str, Any]]:
    """Return all discovered plugins grouped by kind.

    The returned dict has keys ``guardrails`` and ``verifiers``. Each value is
    a mapping from entry-point name to the loaded object.
    """
    plugins: dict[str, dict[str, Any]] = {"guardrails": {}, "verifiers": {}}

    try:
        eps = entry_points()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read entry points: %s", exc)
        return plugins

    for group, target in (
        (_GUARD_ENTRY_POINT_GROUP, "guardrails"),
        (_VERIFIER_ENTRY_POINT_GROUP, "verifiers"),
    ):
        try:
            if hasattr(eps, "select"):
                entries = eps.select(group=group)
            else:
                entries = eps.get(group, [])  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read entry points for %s: %s", group, exc)
            continue

        for entry in entries:
            try:
                plugins[target][entry.name] = entry.load()
            except Exception as exc:
                logger.warning("Failed to load plugin %r from %s: %s", entry.name, group, exc)

    return plugins


def _guardrail_from_value(value: Any) -> Any:
    """Normalize a guardrails plugin entry into a Guardrail instance."""
    from .guardrails import Guardrail

    if isinstance(value, Guardrail):
        return value
    if isinstance(value, type) and issubclass(value, Guardrail):
        return value()  # type: ignore[call-arg]
    raise TypeError(f"Guardrail plugin must be a Guardrail instance or class, got {type(value)}")


def _verifier_from_value(value: Any) -> Any:
    """Normalize a verifier plugin entry into a grounding callable."""
    from .verifier import Verifier

    if callable(value):
        return Verifier(check_fn=value)
    if isinstance(value, Verifier):
        return value
    raise TypeError(f"Verifier plugin must be a callable or Verifier instance, got {type(value)}")


def load_plugins(
    plugins: list[Any] | str | None,
) -> tuple[list[Any], list[Any]]:
    """Resolve plugin references into (guardrails, verifiers).

    ``plugins`` can be:

    - ``None``: no plugins loaded.
    - ``"auto"``: load every discovered plugin.
    - A list mixing entry-point names (strings), Guardrail instances/classes,
      Verifier instances/callables, and module-level factory functions.
    """
    discovered = discover_plugins()
    guardrails: list[Any] = []
    verifiers: list[Any] = []

    if plugins is None:
        return guardrails, verifiers

    if plugins == "auto":
        for value in discovered["guardrails"].values():
            guardrails.append(_guardrail_from_value(value))
        for value in discovered["verifiers"].values():
            verifiers.append(_verifier_from_value(value))
        return guardrails, verifiers

    if isinstance(plugins, str):
        plugins = [plugins]

    for item in plugins:
        if isinstance(item, str):
            if item in discovered["guardrails"]:
                guardrails.append(_guardrail_from_value(discovered["guardrails"][item]))
            elif item in discovered["verifiers"]:
                verifiers.append(_verifier_from_value(discovered["verifiers"][item]))
            else:
                raise ValueError(f"Unknown plugin name: {item!r}")
        else:
            from .guardrails import Guardrail
            from .verifier import Verifier

            if isinstance(item, Guardrail):
                guardrails.append(item)
            elif isinstance(item, type) and issubclass(item, Guardrail):
                guardrails.append(item())  # type: ignore[call-arg]
            elif isinstance(item, Verifier):
                verifiers.append(item)
            elif callable(item):
                verifiers.append(Verifier(check_fn=item))
            else:
                raise TypeError(f"Unsupported plugin value: {item!r}")

    return guardrails, verifiers
