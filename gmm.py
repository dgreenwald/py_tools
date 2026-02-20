import warnings

warnings.warn(
    "Importing from py_tools.gmm is deprecated. "
    "Use py_tools.econometrics.gmm instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.gmm import *  # noqa: F401,F403
