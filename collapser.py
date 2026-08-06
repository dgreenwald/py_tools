import warnings

warnings.warn(
    "Importing from py_tools.collapser is deprecated. "
    "Use py_tools.data.collapser instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.data.collapser import *  # noqa: F401,F403,E402
