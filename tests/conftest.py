"""Shared pytest configuration for the optional slow integration suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked as slow, including real embedding-model integrations",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="slow tests require explicit --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
