import warnings

warnings.warn(
    "Importing from py_tools.config_registry is deprecated. "
    "Use py_tools.config.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.config.registry import *  # noqa: F401,F403
