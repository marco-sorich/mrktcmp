# ---------------------------------------------------------------------------
# strategies/registry.py – explicit plugin registry
#
# Plugins call @register on their class at module load time.  The app
# imports src.strategies (the package __init__.py) at startup, which
# imports every plugin module, which triggers their @register calls.
# ---------------------------------------------------------------------------

from __future__ import annotations

from src.strategies.base import BacktestStrategy

# Maps strategy name (get_name() return value) → strategy class.
_registry: dict[str, type[BacktestStrategy]] = {}


def register(cls: type[BacktestStrategy]) -> type[BacktestStrategy]:
    """Class decorator: add *cls* to the registry under its get_name() key.

    Usage
    -----
    @register
    class MyStrategy(BacktestStrategy):
        ...
    """
    _registry[cls.get_name()] = cls
    return cls


def get_strategy(name: str) -> type[BacktestStrategy]:
    """Return the strategy class registered under *name*.

    Raises KeyError if the name is unknown, listing available strategies in
    the message so callers can surface a helpful error to the user.
    """
    if name not in _registry:
        raise KeyError(
            f"Unknown strategy: {name!r}. Available: {list(_registry)}"
        )
    return _registry[name]


def list_strategies() -> list[str]:
    """Return the names of all registered strategies, in insertion order."""
    return list(_registry)


def get_all_strategy_info() -> list[dict[str, str]]:
    """Return display metadata for all registered strategies, in registration order.

    Each dict contains:
      'name'        – the strategy's unique identifier (same as its registry key).
      'icon'        – Bootstrap icon class name for the GUI (e.g. 'bi-calendar-month').
      'description' – one-sentence human-readable description.
    """
    return [
        {
            'name': name,
            'icon': cls.get_icon(),
            'description': cls.get_description(),
        }
        for name, cls in _registry.items()
    ]
