import warnings

warnings.warn(
    "Importing from py_tools.stata is deprecated. Use py_tools.in_out.stata instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.in_out.stata import *  # noqa: F401,F403,E402
