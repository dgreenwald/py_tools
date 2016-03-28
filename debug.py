import numpy as np
from tabulate import tabulate

def disp(mat):
    if len(mat.shape) == 1:
        print(tabulate(mat[:, np.newaxis], floatfmt='.4f'))
    else:
        print(tabulate(mat, floatfmt='.4f'))
    return None
