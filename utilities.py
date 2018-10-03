from collections import Mapping
import itertools
import numpy as np
from time import time

def as_list(x):

    if not isinstance(x, list):
        return [x]
    else:
        return x

def split(x, lengths, axis=0):
    """Split a numpy matrix or array into pieces with lengths according to input list"""
    indices = [lengths[0]]
    for length in lengths[1:-1]:
        indices.append(indices[-1] + length)
    return np.split(x, indices, axis=axis)

def split_str(string, length):
    """Split a string into two pieces, break occurs at argument"""
    string = str(string)
    return (string[:length], string[length:])

def split_list(x, n):
    """Shortcut to split a list in two"""
    return x[:n], x[n:]

def any2(list_of_items, list_to_check):
    """Check of any on left are in list on right"""
    return any([var in list_to_check for var in list_of_items])

def join_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def check_duplicates(var_list):

    if len(var_list) != len(set(var_list)):
        seen = set()
        duplicate = set()
        for var in var_list:
            if var not in seen:
                seen.add(var)
            else:
                duplicate.add(var)
        print("Repeated variables detected:")
        print(duplicate)
        raise Exception

def swap_all_axes(a, target_axes, count=0):

    if count == len(a.shape):

        return a

    else:

        ix = target_axes.index(count)

        if ix != count:
            new_axes = target_axes.copy()
            a_new = np.swapaxes(a, ix, count)

            new_axes[count] = target_axes[ix]
            new_axes[ix] = target_axes[count] 
        else:
            new_axes = target_axes
            a_new = a
        
        return swap_all_axes(a_new, new_axes, count=(count+1))

def tic():

    return time()

def toc(start, display=True):

    elapsed = time() - start
    print("Time elapsed: {:4.3g} seconds".format(elapsed))
    return elapsed

def log_if_pos(x):
    """
    Compute the log of an array if positive, otherwise return NaNs
    Parameters
    ----------
    x : array-like 
        Array to take log of
    Returns 
    -------
    out: ndarray
        Log if all positive, otherwise NaN
    """

    if np.all(x > 0):
        return np.log(x)
    else:
        return np.nan * np.ones(x.shape)

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def cartesian_matrices(A, B):

    Nx = A.shape[0] * B.shape[0]
    Ny = A.shape[1] * B.shape[1]

    C = np.zeros((Nx, Ny, 2))

    for ii, jj in itertools.product(range(A.shape[0]), range(B.shape[0])):

        kk = B.shape[0] * ii + jj 
        C[kk, :, :] = cartesian((A[ii, :], B[jj, :]))

    return C
