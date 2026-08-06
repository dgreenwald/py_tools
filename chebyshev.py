import warnings

warnings.warn(
    "Importing from py_tools.chebyshev is deprecated. "
    "Use py_tools.numerical.chebyshev instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.numerical.chebyshev import *  # noqa: F401,F403,E402
