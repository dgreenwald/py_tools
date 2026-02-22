import re
from collections import OrderedDict
from collections.abc import Mapping

class MySet(set):
    """Set plus addition operator"""

    def __init__(self, *args, **kwargs):
        set.__init__(self, *args, **kwargs)

    def __add__(self, other):
        return MySet(self.copy() | MySet(other))

    def __radd__(self, other):
        return MySet(self.copy() | MySet(other))

class MyDict(dict):
    """Dict plus addition operator"""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        temp = self.copy()
        temp.update(OrderedDict(other))
        return MyDict(temp)

    def __radd__(self, other):
        temp = OrderedDict(other)
        temp.update(self)
        return MyDict(temp)

class MyOrderedDict(OrderedDict):
    """OrderedDict plus addition operator"""

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        temp = self.copy()
        temp.update(OrderedDict(other))
        return MyOrderedDict(temp)

    def __radd__(self, other):
        temp = OrderedDict(other)
        temp.update(self)
        return MyOrderedDict(temp)

class PresetDict(dict):
    """Preset dict that will not update if key already set"""

    def __init__(self, other=None, verbose=False, **kwargs):
        super().__init__()
        self.verbose=verbose
        self.update(other, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)
        elif self.verbose:
            print("PresetDict: ignoring key '{}'".format(key))

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def overwrite_item(self, key, value):

        super().__setitem__(key, value)
        
    def overwrite_update(self, other):
        
        super().update(other)
        
class UniqueList:
    """List that silently ignores duplicate entries on construction and addition."""

    def __init__(self, iterable=None):
        self.data = []
        if iterable is not None:
            for x in iterable:
                if x not in self.data:
                    self.data.append(x)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    def __add__(self, other):
        unique = UniqueList([x for x in other if x not in self.data])
        return self.data + unique.data

    def __radd__(self, other):
        unique = UniqueList([x for x in other if x not in self.data])
        return self.data + unique.data

    def __iadd__(self, other):
        for x in other:
            if x not in self.data:
                self.data.append(x)
        return self

        
def replace_keys(my_dict, orig, repl):
    """Replace keys in a dict according to a regex pattern"""

    for key in list(my_dict.keys()):
        new_key = re.sub(orig, repl, key)
        my_dict[new_key] = my_dict.pop(key)

    return None

def replace_keys_items(my_dict, orig, repl):
    """Replace keys and string items in a dict according to a regex pattern"""

    for key in list(my_dict.keys()):
        new_key = re.sub(orig, repl, key)
        value = my_dict.pop(key)
        my_dict[new_key] = re.sub(orig, repl, value) if isinstance(value, str) else value
