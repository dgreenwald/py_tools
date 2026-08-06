import warnings

warnings.warn(
    "Importing from py_tools.match is deprecated. Use py_tools.data.match instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.data.match import *  # noqa: F401,F403,E402
