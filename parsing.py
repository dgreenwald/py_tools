import warnings

warnings.warn(
    "Importing from py_tools.parsing is deprecated. Use py_tools.text.parsing instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.text.parsing import *  # noqa: F401,F403,E402
