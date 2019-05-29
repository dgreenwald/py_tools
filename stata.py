import numpy as np
import pandas as pd

def read_coefficients(file_stem=None, file_b=None, file_V=None):
    """Read coefficients from saved mat2txt files"""

    if file_b is None:
        file_b = file_stem + '_b.txt'

    if file_V is None:
        file_V = file_stem + '_V.txt'

    b = pd.read_csv(file_b, sep='\s+').reset_index().drop(columns='index')
    V = pd.read_csv(file_V, sep='\s+')
    
    se_data = np.sqrt(np.diag(V.values))
    # se_names = [name + '_se' for name in V.columns]
    
    se = pd.DataFrame(data=se_data[np.newaxis, :], columns=V.columns)

    return b, V, se
