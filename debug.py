import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def disp(mat):
    """Display in Matlab form"""

    if len(mat.shape) == 1:
        print(tabulate(mat[:, np.newaxis], floatfmt='.4f'))
    else:
        print(tabulate(mat, floatfmt='.4f'))
    return None

def plow(x):
    """Plot and show pandas object"""

    x.dropna().plot()
    plt.show()
    return None
