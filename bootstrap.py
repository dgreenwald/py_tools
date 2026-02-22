import warnings

warnings.warn(
    "Importing from py_tools.bootstrap is deprecated. "
    "Use py_tools.econometrics.bootstrap instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.bootstrap import *  # noqa: F401,F403,E402
