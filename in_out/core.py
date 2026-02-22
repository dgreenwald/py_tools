import numpy as np
import pandas as pd
import os
import pickle
import struct
import zipfile
import json

def load_eigen(filename, dtype='float64'):
    """Load a 2-D array written in the Eigen binary format.

    The file is expected to begin with two 4-byte little-endian integers
    giving the number of rows and columns, followed by the matrix data
    stored in column-major (Fortran) order.

    Parameters
    ----------
    filename : str
        Path to the binary file.
    dtype : str or numpy.dtype, optional
        Data type of the stored values, by default ``'float64'``.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(n_rows, n_cols)`` with the loaded data.
    """
    with open(filename, 'rb') as fid:
        n_rows = struct.unpack('i', fid.read(4))[0]
        n_cols = struct.unpack('i', fid.read(4))[0]
        return np.fromfile(fid, dtype=dtype).reshape(n_cols, n_rows).transpose()

def reshape_eigen(filename, shape, dtype='float64'):
    """Load an Eigen binary file and reshape the data to a custom shape.

    Reads the row/column counts from the file header, verifies that the
    total element count matches ``shape``, and returns the data reshaped
    accordingly.

    Parameters
    ----------
    filename : str
        Path to the binary file written in Eigen format.
    shape : tuple of int
        Desired output shape.  The product of all dimensions must equal
        ``n_rows * n_cols`` stored in the file header.
    dtype : str or numpy.dtype, optional
        Data type of the stored values, by default ``'float64'``.

    Returns
    -------
    numpy.ndarray
        Array reshaped to ``shape``.

    Raises
    ------
    ValueError
        If the total number of elements implied by ``shape`` does not
        match the element count in the file.
    """
    with open(filename, 'rb') as fid:

        n_rows = struct.unpack('i', fid.read(4))[0]
        n_cols = struct.unpack('i', fid.read(4))[0]

        if (n_rows * n_cols) != np.prod(shape):
            raise ValueError(
                f"reshape_eigen: shape {shape} does not match file size {n_rows * n_cols}"
            )

        return np.fromfile(fid, dtype=dtype).reshape(shape)

def save_eigen(vals, filename, dtype='float64'):
    """Save a 2-D array to a binary file in Eigen format.

    Writes a 4-byte integer header containing the number of rows and
    columns, followed by the matrix data in column-major order.

    Parameters
    ----------
    vals : numpy.ndarray
        2-D array to save.  Must have a ``shape`` attribute with exactly
        two elements.
    filename : str
        Destination file path.  The file is created or overwritten.
    dtype : str or numpy.dtype, optional
        Data type to cast values to before writing, by default
        ``'float64'``.

    Returns
    -------
    None
    """
    n_rows, n_cols = vals.shape

    with open(filename, 'wb') as fid:
        fid.write(struct.pack('i', n_rows))
        fid.write(struct.pack('i', n_cols))
        vals.transpose().astype(dtype).tofile(fid)

    return None

def save_pickle(obj, filename):
    """Serialize an object to a file using pickle.

    Parameters
    ----------
    obj : object
        Any Python object that can be pickled.
    filename : str
        Destination file path.  The file is created or overwritten.

    Returns
    -------
    None
    """
    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid)
    return None

def load_pickle(filename):
    """Deserialize an object from a pickle file.

    Parameters
    ----------
    filename : str
        Path to the pickle file.

    Returns
    -------
    object
        The Python object stored in the file.
    """
    with open(filename, 'rb') as fid:
        obj = pickle.load(fid)
    return obj 

def make_dir(path):
    """Create a directory, including any missing parent directories.

    Does nothing if the directory already exists.

    Parameters
    ----------
    path : str
        Directory path to create.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def write_numeric(val, filename, precision='4.3f'):
    """Write a single numeric value to a plain-text file.

    Parameters
    ----------
    val : int or float
        Numeric value to write.
    filename : str
        Destination file path.  The file is created or overwritten.
    precision : str, optional
        Python format specification string used to format *val*, by
        default ``'4.3f'``.

    Returns
    -------
    None
    """
    with open(filename, 'w') as fid:
        fid.write('{0:{1}}'.format(val, precision))
        
    return None

def write_text(string, filename):
    """Write a string to a plain-text file.

    Parameters
    ----------
    string : str
        Text content to write.
    filename : str
        Destination file path.  The file is created or overwritten.

    Returns
    -------
    None
    """
    with open(filename, 'w') as fid:
        fid.write(string)
        
    return None

def write_json(obj, filename, **kwargs):
    """Serialize an object to a JSON file.

    Parameters
    ----------
    obj : object
        JSON-serializable Python object.
    filename : str
        Destination file path.  The file is created or overwritten.
    **kwargs
        Additional keyword arguments forwarded to :func:`json.dump`.

    Returns
    -------
    None
    """
    with open(filename, 'w') as fid:
        json.dump(obj, fid, **kwargs)

    return None

def read_numeric(filename):
    """Read a single numeric value from a plain-text file.

    Parameters
    ----------
    filename : str
        Path to the text file containing the numeric value.

    Returns
    -------
    float
        The numeric value read from the file.
    """
    with open(filename, 'r') as fid:
        val = float(fid.read())
        
    return val

def read_zipped(zip_path, file_name, **kwargs):
    """Read a CSV file from inside a ZIP archive into a DataFrame.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP archive.
    file_name : str
        Name of the CSV file inside the archive.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Contents of the CSV file.
    """
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(file_name) as fid:
            df = pd.read_csv(fid, **kwargs)
            
    return df

def read_json(filename, **kwargs):
    """Deserialize a JSON file into a Python object.

    Parameters
    ----------
    filename : str
        Path to the JSON file.
    **kwargs
        Additional keyword arguments forwarded to :func:`json.load`.

    Returns
    -------
    object
        The Python object decoded from the JSON file.
    """
    with open(filename, 'r') as fid:
        return json.load(fid)
