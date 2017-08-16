import numpy as np

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

