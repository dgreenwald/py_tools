import warnings

warnings.warn(
    "Importing from py_tools.aim is deprecated. Use py_tools.econ.aim instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econ.aim import *  # noqa: F401,F403,E402
