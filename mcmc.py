import warnings

warnings.warn(
    "Importing from py_tools.mcmc is deprecated. Use py_tools.bayesian.mcmc instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.bayesian.mcmc import *  # noqa: F401,F403,E402
