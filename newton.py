import warnings

warnings.warn(
    "Importing from py_tools.newton is deprecated. "
    "Use py_tools.numerical.newton instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.numerical.newton import *  # noqa: F401,F403
