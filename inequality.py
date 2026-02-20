import warnings

warnings.warn(
    "Importing from py_tools.inequality is deprecated. "
    "Use py_tools.stats.inequality instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.stats.inequality import *  # noqa: F401,F403
