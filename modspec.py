import warnings

warnings.warn(
    "Importing from py_tools.modspec is deprecated. "
    "Use py_tools.config.modspec instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.config.modspec import *  # noqa: F401,F403,E402
