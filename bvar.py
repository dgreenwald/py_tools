import warnings

warnings.warn(
    "Importing from py_tools.bvar is deprecated. "
    "Use py_tools.time_series.bvar instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.time_series.bvar import *  # noqa: F401,F403
