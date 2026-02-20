import warnings

warnings.warn(
    "Importing from py_tools.financial is deprecated. "
    "Use py_tools.econ.financial instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econ.financial import *  # noqa: F401,F403
