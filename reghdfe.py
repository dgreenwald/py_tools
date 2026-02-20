import warnings

warnings.warn(
    "Importing from py_tools.reghdfe is deprecated. "
    "Use py_tools.econometrics.reghdfe instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.reghdfe import *  # noqa: F401,F403
