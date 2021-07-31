import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def disp(mat, floatfmt='.4f'):
    """Display in Matlab form"""

    if len(mat.shape) == 1:
        print(tabulate(mat[:, np.newaxis], floatfmt=floatfmt))
    else:
        print(tabulate(mat, floatfmt=floatfmt))
    return None

def plow(x):
    """Plot and show pandas object"""

    x.dropna().plot()
    plt.show()
    return None
