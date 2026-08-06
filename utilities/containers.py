import re
from collections import OrderedDict
from collections.abc import Mapping


class MySet(set):
    """A :class:`set` subclass that supports the ``+`` operator for union.

    Attributes
    ----------
    (inherits all set attributes)
    """

    def __init__(self, *args, **kwargs):
        """Initialize MySet.

        Parameters
        ----------
        *args : positional arguments
            Forwarded to :class:`set`.
        **kwargs : keyword arguments
            Forwarded to :class:`set`.
        """

        set.__init__(self, *args, **kwargs)

    def __add__(self, other):
        """Return the union of this set and *other* as a new MySet.

        Parameters
        ----------
        other : iterable
            Items to union with this set.

        Returns
        -------
        MySet
            Union of ``self`` and ``other``.
        """
        return MySet(self.copy() | MySet(other))

    def __radd__(self, other):
        """Return the union of *other* and this set as a new MySet.

        Parameters
        ----------
        other : iterable
            Items to union with this set.

        Returns
        -------
        MySet
            Union of ``other`` and ``self``.
        """
        return MySet(self.copy() | MySet(other))


class MyDict(dict):
    """A :class:`dict` subclass that supports the ``+`` operator for merging.

    The right-hand operand's keys take precedence on ``self + other``,
    while ``self``'s keys take precedence on ``other + self``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize MyDict.

        Parameters
        ----------
        *args : positional arguments
            Forwarded to :class:`dict`.
        **kwargs : keyword arguments
            Forwarded to :class:`dict`.
        """
        dict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        """Return a new MyDict with *other* merged into a copy of ``self``.

        Parameters
        ----------
        other : mapping or iterable of pairs
            Key-value pairs to merge; *other* keys overwrite ``self`` keys.

        Returns
        -------
        MyDict
            Merged dictionary.
        """
        temp = self.copy()
        temp.update(OrderedDict(other))
        return MyDict(temp)

    def __radd__(self, other):
        """Return a new MyDict with ``self`` merged into a copy of *other*.

        Parameters
        ----------
        other : mapping or iterable of pairs
            Base key-value pairs; ``self`` keys overwrite *other* keys.

        Returns
        -------
        MyDict
            Merged dictionary.
        """
        temp = OrderedDict(other)
        temp.update(self)
        return MyDict(temp)


class MyOrderedDict(OrderedDict):
    """An :class:`OrderedDict` subclass that supports the ``+`` operator for merging.

    The right-hand operand's keys take precedence on ``self + other``,
    while ``self``'s keys take precedence on ``other + self``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize MyOrderedDict.

        Parameters
        ----------
        *args : positional arguments
            Forwarded to :class:`OrderedDict`.
        **kwargs : keyword arguments
            Forwarded to :class:`OrderedDict`.
        """
        OrderedDict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        """Return a new MyOrderedDict with *other* merged into a copy of ``self``.

        Parameters
        ----------
        other : mapping or iterable of pairs
            Key-value pairs to merge; *other* keys overwrite ``self`` keys.

        Returns
        -------
        MyOrderedDict
            Merged ordered dictionary.
        """
        temp = self.copy()
        temp.update(OrderedDict(other))
        return MyOrderedDict(temp)

    def __radd__(self, other):
        """Return a new MyOrderedDict with ``self`` merged into a copy of *other*.

        Parameters
        ----------
        other : mapping or iterable of pairs
            Base key-value pairs; ``self`` keys overwrite *other* keys.

        Returns
        -------
        MyOrderedDict
            Merged ordered dictionary.
        """
        temp = OrderedDict(other)
        temp.update(self)
        return MyOrderedDict(temp)


class PresetDict(dict):
    """A :class:`dict` subclass that silently ignores updates to existing keys.

    Once a key is set, subsequent attempts to set it via ``__setitem__``
    or :meth:`update` are ignored (optionally logged when *verbose* is
    ``True``).  Use :meth:`overwrite_item` or :meth:`overwrite_update`
    to bypass this protection.

    Parameters
    ----------
    other : mapping or iterable of pairs, optional
        Initial key-value pairs. Default is ``None``.
    verbose : bool, optional
        If ``True``, print a message when a duplicate key is ignored.
        Default is ``False``.
    **kwargs : keyword arguments
        Additional initial key-value pairs.

    Attributes
    ----------
    verbose : bool
        Whether to print a message when ignoring duplicate keys.
    """

    def __init__(self, other=None, verbose=False, **kwargs):
        """Initialize PresetDict.

        Parameters
        ----------
        other : mapping or iterable of pairs, optional
            Initial key-value pairs. Default is ``None``.
        verbose : bool, optional
            Print a message when a duplicate key is ignored. Default is
            ``False``.
        **kwargs : keyword arguments
            Additional initial key-value pairs.
        """
        super().__init__()
        self.verbose = verbose
        self.update(other, **kwargs)

    def __setitem__(self, key, value):
        """Set *key* to *value* only if *key* is not already present.

        Parameters
        ----------
        key : hashable
            Dictionary key.
        value : object
            Value to associate with *key*.
        """
        if key not in self:
            super().__setitem__(key, value)
        elif self.verbose:
            print("PresetDict: ignoring key '{}'".format(key))

    def update(self, other=None, **kwargs):
        """Update the dictionary, ignoring keys that are already set.

        Parameters
        ----------
        other : mapping or iterable of pairs, optional
            Key-value pairs to add. Existing keys are silently skipped.
            Default is ``None``.
        **kwargs : keyword arguments
            Additional key-value pairs to add.
        """
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def overwrite_item(self, key, value):
        """Set *key* to *value*, bypassing the preset-protection logic.

        Parameters
        ----------
        key : hashable
            Dictionary key to overwrite.
        value : object
            New value.
        """

        super().__setitem__(key, value)

    def overwrite_update(self, other):
        """Merge *other* into this dict, overwriting any existing keys.

        Parameters
        ----------
        other : mapping
            Key-value pairs to merge unconditionally.
        """

        super().update(other)


class UniqueList:
    """An ordered list that silently discards duplicate entries.

    Duplicates are ignored both during construction and when using ``+=``
    or ``+``.  The underlying storage is a plain :class:`list` accessible
    via the ``data`` attribute.

    Parameters
    ----------
    iterable : iterable, optional
        Initial items.  Duplicates are dropped in order of first
        occurrence.  Default is ``None`` (empty list).

    Attributes
    ----------
    data : list
        Internal list of unique items in insertion order.
    """

    def __init__(self, iterable=None):
        """Initialize UniqueList.

        Parameters
        ----------
        iterable : iterable, optional
            Initial items; duplicates are silently dropped.
            Default is ``None``.
        """
        self.data = []
        if iterable is not None:
            for x in iterable:
                if x not in self.data:
                    self.data.append(x)

    def __repr__(self):
        """Return a string representation of the UniqueList.

        Returns
        -------
        str
            Representation in the form ``UniqueList([...])``.
        """
        return "{}({!r})".format(type(self).__name__, self.data)

    def __len__(self):
        """Return the number of unique items.

        Returns
        -------
        int
            Number of items in the list.
        """
        return len(self.data)

    def __iter__(self):
        """Return an iterator over the unique items.

        Returns
        -------
        iterator
            Iterator over ``self.data``.
        """
        return iter(self.data)

    def __contains__(self, item):
        """Check membership.

        Parameters
        ----------
        item : object
            Item to look for.

        Returns
        -------
        bool
            ``True`` if *item* is in the list.
        """
        return item in self.data

    def __add__(self, other):
        """Concatenate with *other*, excluding items already in ``self``.

        Parameters
        ----------
        other : iterable
            Items to append (duplicates of existing items are dropped).

        Returns
        -------
        list
            Plain list containing ``self.data`` followed by new unique
            items from *other*.
        """
        unique = UniqueList([x for x in other if x not in self.data])
        return self.data + unique.data

    def __radd__(self, other):
        """Concatenate *other* with ``self``, excluding items already in ``self``.

        Parameters
        ----------
        other : iterable
            Left-hand items; items duplicated in ``self`` are dropped.

        Returns
        -------
        list
            Plain list containing ``self.data`` followed by new unique
            items from *other*.
        """
        unique = UniqueList([x for x in other if x not in self.data])
        return self.data + unique.data

    def __iadd__(self, other):
        """Append unique items from *other* in-place.

        Parameters
        ----------
        other : iterable
            Items to add; duplicates of existing items are silently ignored.

        Returns
        -------
        UniqueList
            ``self`` after appending new unique items.
        """
        for x in other:
            if x not in self.data:
                self.data.append(x)
        return self


def replace_keys(my_dict, orig, repl):
    """Replace dictionary keys in-place using a regex substitution.

    Parameters
    ----------
    my_dict : dict
        Dictionary whose keys will be modified in-place.
    orig : str
        Regex pattern to search for in each key.
    repl : str
        Replacement string (may contain back-references).

    Returns
    -------
    None
        The dictionary is modified in-place; nothing is returned.
    """

    for key in list(my_dict.keys()):
        new_key = re.sub(orig, repl, key)
        my_dict[new_key] = my_dict.pop(key)

    return None


def replace_keys_items(my_dict, orig, repl):
    """Replace dictionary keys and string values in-place using a regex substitution.

    Parameters
    ----------
    my_dict : dict
        Dictionary to modify in-place.
    orig : str
        Regex pattern to search for in each key and in string values.
    repl : str
        Replacement string (may contain back-references).

    Returns
    -------
    None
        The dictionary is modified in-place; nothing is returned.
    """

    for key in list(my_dict.keys()):
        new_key = re.sub(orig, repl, key)
        value = my_dict.pop(key)
        my_dict[new_key] = (
            re.sub(orig, repl, value) if isinstance(value, str) else value
        )
