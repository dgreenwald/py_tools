import warnings

warnings.warn(
    "Importing from py_tools.vector_autoregression is deprecated. "
    "Use py_tools.time_series.var instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.time_series.var import *  # noqa: F401,F403
