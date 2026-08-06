import warnings

warnings.warn(
    "Importing from py_tools.state_space is deprecated. "
    "Use py_tools.time_series.state_space instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.time_series.state_space import *  # noqa: F401,F403,E402
