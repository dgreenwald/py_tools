import numpy as np
import pandas as pd
import os
import pickle
import struct
import zipfile

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
    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid)
    return None

def load_pickle(filename):
    with open(filename, 'rb') as fid:
        obj = pickle.load(fid)
    return obj 

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def write_text(string, filename):
    
    with open(filename, 'w') as fid:
        fid.write(string)
        
    return None

def write_numeric(val, filename, precision='4.3f'):
    
    with open(filename, 'w') as fid:
        fid.write('{0:{1}}'.format(val, precision))
        
    return None

def read_numeric(filename):
    
    with open(filename, 'r') as fid:
        val = float(fid.read())
        
    return val

def read_zipped(zip_path, file_name, **kwargs):
    
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(file_name) as fid:
            df = pd.read_csv(fid, **kwargs)
            
    return df
