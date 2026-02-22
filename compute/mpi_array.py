import numpy as np
from mpi4py import MPI

def rank():
    """Return the MPI rank of the calling process.

    Returns
    -------
    int
        Rank of this process in ``MPI.COMM_WORLD`` (0-based).
    """
    
    return MPI.COMM_WORLD.Get_rank()

def disp(mesg, flush=True, **kwargs):
    """Print a message from MPI rank 0 only.

    Parameters
    ----------
    mesg : str
        Message to print.
    flush : bool, optional
        Whether to flush stdout immediately.  Default is ``True``.
    **kwargs
        Additional keyword arguments forwarded to :func:`print`.
    """
    
    if rank() == 0:
        print(mesg, flush=flush, **kwargs)

def time():
    """Return the current wall-clock time as reported by MPI.

    Returns
    -------
    float
        Elapsed wall-clock time in seconds (``MPI.Wtime()``).
    """

    return MPI.Wtime()

def barrier():
    """Block until all MPI processes have reached this point.

    Wraps ``MPI.Comm.Barrier(MPI.COMM_WORLD)``.
    """
    
    MPI.Comm.Barrier(MPI.COMM_WORLD)

def initialize(x, fake=False):
    """Create an :class:`MPIArray` (or :class:`FakeMPIArray`) and scatter *x*.

    Parameters
    ----------
    x : ndarray
        Root array to distribute across MPI processes.
    fake : bool, optional
        If ``True``, wrap *x* in a :class:`FakeMPIArray` so the code runs
        without a real MPI environment.  Default is ``False``.

    Returns
    -------
    x_mpi : MPIArray or FakeMPIArray
        Distributed array object.
    x_loc : ndarray
        Local slice of *x* for the calling process.
    """

    if fake:
        x_mpi = FakeMPIArray(x)
    else:
        x_mpi = MPIArray(x)

    x_loc = x_mpi.get_local_data()

    return x_mpi, x_loc

def finalize(x_mpi, x_loc):
    """Write local results back and retrieve the full array on rank 0.

    Parameters
    ----------
    x_mpi : MPIArray or FakeMPIArray
        Distributed array object created by :func:`initialize`.
    x_loc : ndarray
        Updated local slice produced by the calling process.

    Returns
    -------
    ndarray or None
        Full gathered array on rank 0; ``None`` on all other ranks when
        using a real :class:`MPIArray`.
    """

    x_mpi.set_local_data(x_loc)
    return x_mpi.get_root_data()

#class MPIPrinter:
#    """Object that holds rank and prints only for rank 0"""
#
#    def __init__(self, flush=True):
#
#        self.flush = flush
#        self.rank = MPI.COMM_WORLD.Get_rank()
#
#    def print(self, mesg, flush=None):
#
#        if self.rank == 0:
#            if flush is None:
#                flush = self.flush
#
#            print(mesg, flush=flush)

class MPIArray:
    """Array that automatically does MPI sharing.

    Distributes the leading dimension of a NumPy array across MPI ranks
    using scatter/gather operations.  Padding is added so that every rank
    holds the same number of rows.

    Parameters
    ----------
    root_data : ndarray, optional
        Full array held on rank 0 to be scattered to all ranks.
        Mutually exclusive with *root_shape*.
    root_shape : tuple of int, optional
        Shape of the full array when *root_data* is not provided.
    dtype : str or numpy dtype, optional
        Data type for allocation when *root_data* is not provided.
        Default is ``'float64'``.
    scatter : bool, optional
        If ``True`` (default), scatter *root_data* (or allocate and
        scatter zeros) immediately on construction.

    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (``COMM_WORLD``).
    size : int
        Total number of MPI processes.
    rank : int
        Rank of the calling process.
    root_shape : list of int
        Shape of the full (unpadded) array.
    nrow_root : int
        Number of rows in the full array.
    npad_local : int
        Number of rows held by each rank (including padding).
    npad_root : int
        Total padded number of rows (``npad_local * size``).
    root_data : ndarray or None
        Padded root array, allocated only on rank 0.
    local_data : ndarray
        Local padded data for the calling rank.
    ix_root : ndarray of bool or None
        Boolean mask of length ``npad_root`` indicating which rows are
        real (not padding) in the root array.  ``None`` on non-root ranks.
    ix_local : ndarray of bool
        Boolean mask of length ``npad_local`` indicating which rows are
        real (not padding) in the local buffer.
    """

    def __init__(self, root_data=None, root_shape=None, dtype='float64',
                 scatter=True):

        # Get info from array passed in
        if root_data is not None:
            root_shape = root_data.shape
            self.dtype = root_data.dtype
        else:
            self.dtype = dtype

        # Set up MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Data shape
        self.root_shape = list(root_shape)
        self.nrow_root = self.root_shape[0]
        self.npad_local = ((self.nrow_root - 1) // self.size) + 1
        self.npad_root = self.npad_local * self.size # padded version

        if len(self.root_shape) == 1:
            # One-index vector
            self.ncol = 1
            self.vec_flag = 1
        else:
            # Multidimensional array
            self.higher_dims = self.root_shape[1:]
            self.ncol = int(np.prod(self.higher_dims))
            self.vec_flag = 0

        # Allocate arrays
        if self.rank == 0:
            self.root_data = np.empty((self.npad_root, self.ncol), dtype=self.dtype)
        else:
            self.root_data = None

        self.local_data = np.empty((self.npad_local, self.ncol), dtype=self.dtype) 

        # Allocate index for padding
        if self.rank == 0:
            self.ix_root = np.ones(self.npad_root, dtype=bool)
            self.ix_root[self.nrow_root:] = False
        else:
            self.ix_root = None

        self.ix_local = np.ones(self.npad_local, dtype=bool)

        # Initialize data (if passed)
        if root_data is not None:
            self.set_root_data(root_data, scatter=scatter)
        elif scatter:
            self.scatter()
            
        # Scatter data
        # if scatter: self.scatter()

    def set_root_data(self, data, scatter=True):
        """Copy *data* into the root array and optionally scatter it.

        Parameters
        ----------
        data : ndarray
            New data for the root array.  Only used on rank 0.
        scatter : bool, optional
            If ``True`` (default), scatter the data to all ranks after
            copying.
        """

        if self.rank == 0:
            self.root_data[self.ix_root, :] = data.reshape((self.nrow_root, self.ncol))

        if scatter: self.scatter()

    def set_local_data(self, data, gather=True):
        """Copy *data* into the local buffer and optionally gather it.

        Parameters
        ----------
        data : ndarray
            Updated local data (unpadded) for the calling rank.
        gather : bool, optional
            If ``True`` (default), gather all local data back to rank 0
            after copying.
        """

        self.local_data[self.ix_local, :] = data.reshape((self.nrow_local, self.ncol))
        if gather: self.gather()

    def get_root_data(self, gather=False):
        """Retrieve the full (unpadded) array on rank 0.

        Parameters
        ----------
        gather : bool, optional
            If ``True``, gather local data to rank 0 before returning.
            Default is ``False``.

        Returns
        -------
        ndarray or None
            Full array on rank 0; ``None`` on all other ranks.
        """

        if gather: self.gather()

        if self.rank == 0:

            out_data = self.root_data[self.ix_root, :]

            if self.vec_flag:
                return out_data.ravel()
            else:
                return out_data.reshape(self.root_shape)

        else:

            return None

    def get_local_data(self, scatter=False):
        """Retrieve the local (unpadded) slice for the calling rank.

        Parameters
        ----------
        scatter : bool, optional
            If ``True``, scatter root data to all ranks before returning.
            Default is ``False``.

        Returns
        -------
        ndarray
            Local data array (shape matches the original higher dimensions
            for multi-dimensional arrays, or a 1-D vector for 1-D arrays).
        """

        if scatter: self.scatter()

        out_data = self.local_data[self.ix_local, :]

        if self.vec_flag:
            return out_data.ravel()
        else:
            return out_data.reshape(self.local_shape)

    def scatter(self):
        """Scatter root data from rank 0 to all ranks.

        Distributes :attr:`root_data` and :attr:`ix_root` to all ranks,
        populating :attr:`local_data` and :attr:`ix_local` via
        ``MPI.Comm.Scatter``.  Also updates ``nrow_local`` (and
        ``local_shape`` for multi-dimensional arrays).
        """

        self.comm.Scatter(self.root_data, self.local_data, root=0) 
        self.comm.Scatter(self.ix_root, self.ix_local, root=0)

        # Update number of local rows
        self.nrow_local = np.sum(self.ix_local)
        if not self.vec_flag:
            self.local_shape = [self.nrow_local] + self.higher_dims

    def gather(self):
        """Gather local data from all ranks to rank 0.

        Collects :attr:`local_data` and :attr:`ix_local` from all ranks
        into :attr:`root_data` and :attr:`ix_root` via
        ``MPI.Comm.Gather``.
        """

        self.comm.Gather(self.local_data, self.root_data, root=0)
        self.comm.Gather(self.ix_local, self.ix_root, root=0)

class FakeMPIArray:
    """Regular numpy array with the same interface as :class:`MPIArray`.

    Provides identical ``set_*`` / ``get_*`` / ``scatter`` / ``gather``
    methods so that code written for :class:`MPIArray` runs in a
    single-process environment without any MPI dependency.

    Parameters
    ----------
    root_data : ndarray, optional
        Data to wrap.  Copied by default.
    root_shape : tuple of int, optional
        Shape to allocate when *root_data* is not provided.
    dtype : str or numpy dtype, optional
        Data type for allocation.  Default is ``'float64'``.
    copy : bool, optional
        If ``True`` (default), copy *root_data* on construction.
    **kwargs
        Accepted but ignored (for API compatibility with
        :class:`MPIArray`).

    Attributes
    ----------
    data : ndarray
        The underlying array.
    """

    def __init__(self, root_data=None, root_shape=None, dtype='float64',
                 copy=True, **kwargs):

        if root_data is not None:
            if copy:
                self.data = root_data.copy()
            else:
                self.data = root_data
        else:
            assert (root_shape is not None)
            self.data = np.empty(root_shape, dtype=dtype)

    def set_root_data(self, data, copy=True, **kwargs):
        """Replace the underlying array with *data*.

        Parameters
        ----------
        data : ndarray
            New data.
        copy : bool, optional
            If ``True`` (default), store a copy of *data*.
        **kwargs
            Accepted but ignored.
        """

        if copy:
            self.data = data.copy()
        else:
            self.data = data

    def set_local_data(self, data, copy=True, **kwargs):
        """Replace the underlying array with *data* (local alias).

        Parameters
        ----------
        data : ndarray
            New data.
        copy : bool, optional
            If ``True`` (default), store a copy of *data*.
        **kwargs
            Accepted but ignored.
        """

        if copy:
            self.data = data.copy()
        else:
            self.data = data

    def get_root_data(self, **kwargs):
        """Return the underlying array (root alias).

        Returns
        -------
        ndarray
        """
        return self.data

    def get_local_data(self, **kwargs):
        """Return the underlying array (local alias).

        Returns
        -------
        ndarray
        """
        return self.data

    def scatter(self):
        """No-op scatter (single-process stub).

        Returns
        -------
        None
        """
        return None

    def gather(self):
        """No-op gather (single-process stub).

        Returns
        -------
        None
        """
        return None
