import warnings

warnings.warn(
    "Importing from py_tools.mpi_array is deprecated. "
    "Use py_tools.compute.mpi_array instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.compute.mpi_array import *  # noqa: F401,F403,E402
