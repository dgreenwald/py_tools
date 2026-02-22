import os
import itertools
import numpy as np
from time import perf_counter

def as_list(x):
    """Wrap a non-list value in a list, or return lists unchanged.

    Parameters
    ----------
    x : object
        Value to ensure is a list.

    Returns
    -------
    list
        ``[x]`` if *x* is not already a list, otherwise *x* unchanged.
    """

    if not isinstance(x, list):
        return [x]
    else:
        return x

def split(x, lengths, axis=0):
    """Split a NumPy array into pieces with specified lengths.

    Parameters
    ----------
    x : ndarray
        Array to split.
    lengths : sequence of int
        Length of each resulting piece along *axis*.
    axis : int, optional
        Axis along which to split. Default is 0.

    Returns
    -------
    list of ndarray
        List of sub-arrays with the requested lengths.
    """
    indices = [lengths[0]]
    for length in lengths[1:-1]:
        indices.append(indices[-1] + length)
    return np.split(x, indices, axis=axis)

def split_str(string, length):
    """Split a string into two substrings at a given position.

    Parameters
    ----------
    string : str
        String to split (converted via ``str()`` if not already a string).
    length : int
        Index at which the break occurs; first piece is ``string[:length]``.

    Returns
    -------
    tuple of str
        ``(string[:length], string[length:])``
    """
    string = str(string)
    return (string[:length], string[length:])

def split_list(x, n):
    """Split a list into two parts at index *n*.

    Parameters
    ----------
    x : list
        List to split.
    n : int
        Split index; first part is ``x[:n]``, second is ``x[n:]``.

    Returns
    -------
    tuple of list
        ``(x[:n], x[n:])``
    """
    return x[:n], x[n:]

def any2(list_of_items, list_to_check):
    """Check whether any item from one list appears in another.

    Parameters
    ----------
    list_of_items : iterable
        Items to search for.
    list_to_check : iterable
        Collection to search within.

    Returns
    -------
    bool
        ``True`` if at least one item from *list_of_items* is in
        *list_to_check*, otherwise ``False``.
    """
    return any([var in list_to_check for var in list_of_items])

def join_lists(list_of_lists):
    """Flatten a list of lists into a single list.

    Parameters
    ----------
    list_of_lists : iterable of iterable
        Iterable whose elements are themselves iterable.

    Returns
    -------
    list
        All elements from each sub-iterable concatenated in order.
    """
    return list(itertools.chain.from_iterable(list_of_lists))

def check_duplicates(var_list):
    """Raise an exception if a list contains duplicate entries.

    Parameters
    ----------
    var_list : list
        List of values to check for duplicates.

    Raises
    ------
    Exception
        If any value appears more than once, with the duplicate values
        included in the message.
    """

    if len(var_list) != len(set(var_list)):
        seen = set()
        duplicate = set()
        for var in var_list:
            if var not in seen:
                seen.add(var)
            else:
                duplicate.add(var)
        raise Exception("Repeated variables detected: {}".format(duplicate))

def swap_all_axes(a, target_axes, count=0):
    """Recursively reorder array axes to match a target permutation.

    Parameters
    ----------
    a : ndarray
        Array whose axes are to be reordered.
    target_axes : list of int
        Desired axis ordering expressed as a permutation of
        ``range(a.ndim)``.
    count : int, optional
        Current recursion depth / axis index being placed. Default is 0.

    Returns
    -------
    ndarray
        Array with axes reordered according to *target_axes*.
    """

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
    """Return the current value of the performance counter.

    Returns
    -------
    float
        Current time in fractional seconds, suitable for passing to
        :func:`toc`.
    """

    return perf_counter()

def toc(start, display=True):
    """Print and return elapsed time since *start*.

    Parameters
    ----------
    start : float
        Start time returned by :func:`tic`.
    display : bool, optional
        If ``True``, print the elapsed time. Default is ``True``.

    Returns
    -------
    float
        Elapsed time in seconds.
    """

    elapsed = perf_counter() - start
    print("Time elapsed: {:4.3g} seconds".format(elapsed))
    return elapsed

def timer(func):
    """Decorator that prints the execution time of a function.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        Wrapped function that times its own execution and prints the
        elapsed time via :func:`toc`.
    """
    def wrapper(*args, **kwargs):
        """Execute *func*, printing elapsed time afterwards."""
        start = tic()
        result = func(*args, **kwargs)
        toc(start)
        return result
    return wrapper

def log_if_pos(x):
    """Compute the element-wise log of an array, or return NaNs if any value is non-positive.

    Parameters
    ----------
    x : array-like
        Array to take the log of.

    Returns
    -------
    ndarray
        ``np.log(x)`` when all elements of *x* are strictly positive,
        otherwise an array of NaNs with the same shape as *x*.
    """

    if np.all(x > 0):
        return np.log(x)
    else:
        return np.nan * np.ones(x.shape)

def cartesian(arrays, out=None):
    """Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray, optional
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape ``(M, len(arrays))`` containing all
        combinations formed from the input arrays.

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
    """Compute the row-wise Cartesian product of two 2-D matrices.

    For each pair of rows ``(A[i, :], B[j, :])``, store the combined
    Cartesian product as a single row in the output tensor.

    Parameters
    ----------
    A : ndarray, shape (Na, Ka)
        First matrix.
    B : ndarray, shape (Nb, Kb)
        Second matrix.

    Returns
    -------
    C : ndarray, shape (Na * Nb, Ka * Kb, 2)
        Tensor where ``C[Na*i + j, :, :]`` is the Cartesian product of
        ``A[i, :]`` and ``B[j, :]``.
    """

    Nx = A.shape[0] * B.shape[0]
    Ny = A.shape[1] * B.shape[1]

    C = np.zeros((Nx, Ny, 2))

    for ii, jj in itertools.product(range(A.shape[0]), range(B.shape[0])):

        kk = B.shape[0] * ii + jj 
        C[kk, :, :] = cartesian((A[ii, :], B[jj, :]))

    return C

def get_env(name, default, prefix='', upper=True, dtype=None, no_underscore=False):
    """Read a value from an environment variable with an optional prefix.

    Parameters
    ----------
    name : str
        Base name of the environment variable.
    default : object
        Value to return when the variable is not set.
    prefix : str, optional
        Prefix prepended to *name* (separated by ``'_'`` unless
        *no_underscore* is ``True``). Default is ``''``.
    upper : bool, optional
        Convert the full variable name to upper-case before lookup.
        Default is ``True``.
    dtype : callable, optional
        If provided, the retrieved string value is cast via ``dtype(val)``
        before being returned. Default is ``None`` (no conversion).
    no_underscore : bool, optional
        If ``True``, do not insert an underscore between *prefix* and
        *name*. Default is ``False``.

    Returns
    -------
    object
        The environment variable value (optionally cast via *dtype*), or
        *default* if the variable is not set.
    """

    if (prefix != '') and (not no_underscore) and (prefix[-1] != '_'):
        prefix += '_'

    fullname = prefix + name
    if upper:
        fullname = fullname.upper()

    val = os.environ.get(fullname, default)

    if dtype is not None:
        val = dtype(val)

    return val
