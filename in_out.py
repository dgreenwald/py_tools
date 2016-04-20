import numpy as np
import os
import pickle
import struct
from tabulate import tabulate

def load_eigen(filename, dtype='float64'):

    with open(filename, 'rb') as fid:
        n_rows = struct.unpack('i', fid.read(4))[0]
        n_cols = struct.unpack('i', fid.read(4))[0]
        return np.fromfile(fid, dtype=dtype).reshape(n_cols, n_rows).transpose()

def reshape_eigen(filename, shape, dtype='float64'):

    with open(filename, 'rb') as fid:

        n_rows = struct.unpack('i', fid.read(4))[0]
        n_cols = struct.unpack('i', fid.read(4))[0]

        if (n_rows * n_cols) != np.prod(shape):
            print("reshape_eigen: shape does not match size of file")
            raise Exception

        return np.fromfile(fid, dtype=dtype).reshape(shape)

def save_eigen(vals, filename, dtype='float64'):

    n_rows, n_cols = vals.shape

    with open(filename, 'wb') as fid:
        fid.write(struct.pack('i', n_rows))
        fid.write(struct.pack('i', n_cols))
        vals.transpose().astype(dtype).tofile(fid)

    return None

def save_pickle(obj, filename):
    fullname = '/home/dan/research/frm/save/' + filename + '.pkl'
    with open(fullname, 'wb') as fid:
        pickle.dump(obj, fid)
    return None

def load_pickle(obj, filename):
    fullname = '/home/dan/research/frm/save/' + filename + '.pkl'
    with open(fullname, 'rb') as fid:
        pickle.load(obj, fid)
    return None

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def print_table(table, headers, outfile=None):
    tabulated = tabulate(table, headers, floatfmt='4.3f')

    if outfile is None:
        print(tabulated)
    else:
        with open(outfile, 'w') as fid:
            fid.write(tabulated)

    return None

def write_tabular(fid, table, headers=None, alignment=None, booktabs=True):
    """Write table to latex""" 
    fid.write('\\begin{tabular}')
    if alignment is None:
        alignment = 'l' + len(table[0]) * 'c'

    fid.write('\\toprule\n')
