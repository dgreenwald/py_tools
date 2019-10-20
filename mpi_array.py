import numpy as np
from mpi4py import MPI

def rank():
    
    return MPI.COMM_WORLD.Get_rank()

def disp(mesg, flush=True, **kwargs):
    
    if rank() == 0:
        print(mesg, flush=flush, **kwargs)

def time():

    return MPI.Wtime()

def barrier():
    
    MPI.Comm.Barrier(MPI.COMM_WORLD)

def initialize(x, fake=False):

    if fake:
        x_mpi = FakeMPIArray(x)
    else:
        x_mpi = MPIArray(x)

    x_loc = x_mpi.get_local_data()

    return x_mpi, x_loc

def finalize(x_mpi, x_loc):

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
    """Array that automatically does MPI sharing"""

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

        if self.rank == 0:
            self.root_data[self.ix_root, :] = data.reshape((self.nrow_root, self.ncol))

        if scatter: self.scatter()

    def set_local_data(self, data, gather=True):

        self.local_data[self.ix_local, :] = data.reshape((self.nrow_local, self.ncol))
        if gather: self.gather()

    def get_root_data(self, gather=False):

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

        if scatter: self.scatter()

        out_data = self.local_data[self.ix_local, :]

        if self.vec_flag:
            return out_data.ravel()
        else:
            return out_data.reshape(self.local_shape)

    def scatter(self):

        self.comm.Scatter(self.root_data, self.local_data, root=0) 
        self.comm.Scatter(self.ix_root, self.ix_local, root=0)

        # Update number of local rows
        self.nrow_local = np.sum(self.ix_local)
        if not self.vec_flag:
            self.local_shape = [self.nrow_local] + self.higher_dims

    def gather(self):

        self.comm.Gather(self.local_data, self.root_data, root=0)
        self.comm.Gather(self.ix_local, self.ix_root, root=0)

class FakeMPIArray:
    """Regular numpy array that has the same i/o functions"""

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

        if copy:
            self.data = data.copy()
        else:
            self.data = data

    def set_local_data(self, data, copy=True, **kwargs):

        if copy:
            self.data = data.copy()
        else:
            self.data = data

    def get_root_data(self, **kwargs): return self.data
    def get_local_data(self, **kwargs): return self.data

    def scatter(self): return None
    def gather(self): return None
