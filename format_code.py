import warnings

warnings.warn(
    "Importing from py_tools.format_code is deprecated. "
    "Use py_tools.text.format_code instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.text.format_code import *  # noqa: F401,F403,E402
