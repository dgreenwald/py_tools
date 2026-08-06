import numpy as np
import pandas as pd


def read_coefficients(file_stem=None, file_b=None, file_V=None, to_numpy=False):
    """Read regression coefficients saved by Stata's ``mat2txt`` command.

    At least one of *file_stem* or both *file_b* and *file_V* must be
    provided.  When *file_stem* is given the function derives the file
    names as ``{file_stem}_b.txt`` and ``{file_stem}_V.txt``.

    Parameters
    ----------
    file_stem : str, optional
        Common stem used to derive *file_b* and *file_V* when those are
        not explicitly supplied.
    file_b : str, optional
        Path to the text file containing the coefficient vector.
    file_V : str, optional
        Path to the text file containing the variance-covariance matrix.
    to_numpy : bool, optional
        If ``True``, return plain NumPy arrays instead of DataFrames, by
        default ``False``.

    Returns
    -------
    b : pandas.DataFrame or numpy.ndarray
        Coefficient estimates.  Shape ``(1, k)`` as a DataFrame or
        ``(k,)`` as a 1-D array when *to_numpy* is ``True``.
    V : pandas.DataFrame or numpy.ndarray
        Variance-covariance matrix.  Shape ``(k, k)``.
    se : pandas.DataFrame or numpy.ndarray
        Standard errors derived from the diagonal of *V*.  Shape
        ``(1, k)`` as a DataFrame or ``(k,)`` as a 1-D array when
        *to_numpy* is ``True``.
    """

    if file_b is None:
        file_b = file_stem + "_b.txt"

    if file_V is None:
        file_V = file_stem + "_V.txt"

    b = pd.read_csv(file_b, sep=r"\s+").reset_index().drop(columns="index")
    V = pd.read_csv(file_V, sep=r"\s+")

    se_data = np.sqrt(np.diag(V.values))
    se = pd.DataFrame(data=se_data[np.newaxis, :], columns=V.columns)

    if to_numpy:
        b = b.values.ravel()
        V = V.values
        se = se.values.ravel()

    return b, V, se
