from collections import Mapping
import itertools
import numpy as np

def as_list(x):

    if not isinstance(x, list):
        return [x]
    else:
        return x

def split(x, lengths, axis=0):
    """Split a numpy matrix or array into pieces with lengths according to input list"""
    indices = [lengths[0]]
    for length in lengths[1:-1]:
        indices.append(indices[-1] + length)
    return np.split(x, indices, axis=axis)

def split_str(string, length):
    """Split a string into two pieces, break occurs at argument"""
    string = str(string)
    return (string[:length], string[length:])

def split_list(x, n):
    """Shortcut to split a list in two"""
    return x[:n], x[n:]

def any2(list_of_items, list_to_check):
    """Check of any on left are in list on right"""
    return any([var in list_to_check for var in list_of_items])

def join_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

class PresetDict(dict):
    """dict that does not update if there is already a key present"""

    def __init__(self, other=None, quiet=False, **kwargs):
        super().__init__()
        self.quiet=quiet
        self.update(other, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)
        elif not self.quiet:
            print("PresetDict: ignoring key '{}'".format(key))

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

def swap_all_axes(a, target_axes, count=0):

    if count == len(a.shape):

        return a

    else:

        ix = target_axes.index(count)

        if ix != count:
            new_axes = target_axes.copy()
            a_new = np.swapaxes(a, ix, count)

            new_axes[count] = target_axes[ix]
            new_axes[ix] = target_axes[count] 
        else:
            new_axes = target_axes
            a_new = a
        
        return swap_all_axes(a_new, new_axes, count=(count+1))

