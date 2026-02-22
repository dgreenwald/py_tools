import warnings

warnings.warn(
    "Importing from py_tools.flatten is deprecated. Use py_tools.text.flatten instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.text.flatten import *  # noqa: F401,F403,E402
