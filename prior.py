import warnings

warnings.warn(
    "Importing from py_tools.prior is deprecated. Use py_tools.bayesian.prior instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.bayesian.prior import *  # noqa: F401,F403,E402
