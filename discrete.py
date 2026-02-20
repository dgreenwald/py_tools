import warnings

warnings.warn(
    "Importing from py_tools.discrete is deprecated. "
    "Use py_tools.econ.discrete instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econ.discrete import *  # noqa: F401,F403
