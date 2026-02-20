import warnings

warnings.warn(
    "Importing from py_tools.containers is deprecated. "
    "Use py_tools.utilities.containers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.utilities.containers import *  # noqa: F401,F403
