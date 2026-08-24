"""Tests for py_tools.config.registry."""

from py_tools.config.registry import ConfigRegistry


def test_get_returns_independent_deep_copies():
    registry = ConfigRegistry(
        {
            "base": {
                "settings": {"threshold": 0.5},
                "labels": ["baseline"],
            },
            "variant": {
                "inherits_from": "base",
                "settings": {"enabled": True},
            },
        }
    )

    first = registry.get("variant")
    first["settings"]["threshold"] = 0.9
    first["labels"].append("mutated")

    second = registry.get("variant")

    assert second == {
        "settings": {"threshold": 0.5, "enabled": True},
        "labels": ["baseline"],
    }
    assert first is not second
    assert first["settings"] is not second["settings"]
    assert first["labels"] is not second["labels"]
