import warnings

warnings.warn(
    "Importing from py_tools.estimation is deprecated. "
    "Use py_tools.econometrics.estimation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.estimation import *  # noqa: F401,F403,E402
