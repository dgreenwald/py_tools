"""Top-level package for py_tools.

This module exposes a small, stable set of submodules via lazy imports.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py_tools")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = (
    "data",
    "datasets",
    "numerical",
    "plot",
    "state_space",
    "stats",
    "time_series",
    "utilities",
)


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
