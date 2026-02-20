import warnings

warnings.warn(
    "Importing from py_tools.local_projections is deprecated. "
    "Use py_tools.econometrics.local_projections instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.local_projections import *  # noqa: F401,F403
