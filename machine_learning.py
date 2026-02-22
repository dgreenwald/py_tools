import warnings

warnings.warn(
    "Importing from py_tools.machine_learning is deprecated. "
    "Use py_tools.econometrics.machine_learning instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.econometrics.machine_learning import *  # noqa: F401,F403,E402
