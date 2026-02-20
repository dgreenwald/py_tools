import warnings

warnings.warn(
    "Importing from py_tools.kalman is deprecated. "
    "Use py_tools.time_series.kalman instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.time_series.kalman import *  # noqa: F401,F403
