import warnings

warnings.warn(
    "Importing from py_tools.hidden_markov is deprecated. "
    "Use py_tools.time_series.hidden_markov instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.time_series.hidden_markov import *  # noqa: F401,F403
