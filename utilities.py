import numpy as np

def split(x, lengths, axis=0):
    indices = [lengths[0]]
    for length in lengths[1:-1]:
        indices.append(indices[-1] + length)
    return np.split(x, indices, axis=axis)

def any2(list_of_items, list_to_check):
    return any([var in list_to_check for var in list_of_items])
