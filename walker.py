import warnings

warnings.warn(
    "Importing from py_tools.walker is deprecated. "
    "Use py_tools.stats.walker instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.stats.walker import *  # noqa: F401,F403
