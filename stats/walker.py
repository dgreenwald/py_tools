#!/usr/bin/env python

from numpy import array, ndarray, ones, where
from numpy.random import random, randint

__author__ = "Tamas Nepusz, Denis Bzowy"
__version__ = "27jul2011"

class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probabilities.

    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/

    Parameters
    ----------
    weights : array_like
        Probability weights for each item.  The weights may be in any order
        and do not need to sum to 1.
    keys : array_like, optional
        Labels associated with each weight.  If provided, :meth:`random`
        returns elements from ``keys`` instead of integer indices.

    Attributes
    ----------
    n : int
        Number of items.
    keys : numpy.ndarray or None
        Array of item labels, or ``None`` when no keys were provided.
    prob : numpy.ndarray
        Normalised Walker probability table.
    inx : numpy.ndarray
        Walker alias index table.
    """
    
    def __init__(self, weights, keys=None):
        """Build the Walker tables ``prob`` and ``inx`` for calls to :meth:`random`.

        Parameters
        ----------
        weights : array_like
            Probability weights.  The weights can be in any order and do not
            need to sum to 1.
        keys : array_like, optional
            Labels for each weight entry.  When provided, they are stored as
            a NumPy array and returned by :meth:`random`.

        Raises
        ------
        ValueError
            If ``weights`` is not a 1-D array.
        """
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = array(keys)

        if isinstance(weights, (list, tuple)):
            weights = array(weights, dtype=float)
        elif isinstance(weights, ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -ones(n, dtype=int)
        short = where(weights < 1)[0].tolist()
        long = where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                short.append( k )
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Return random integers or keys with probabilities proportional to the weights.

        Parameters
        ----------
        count : int, optional
            Number of samples to draw.  If ``None`` (default), a single
            integer or key is returned as a scalar.  Otherwise, a NumPy
            array of length ``count`` is returned.

        Returns
        -------
        int, object, or numpy.ndarray
            A single sampled index (or key when ``keys`` was provided) when
            ``count`` is ``None``, or a NumPy array of ``count`` samples.
        """
        if count is None:
            u = random()
            j = randint(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = random(count)
        j = randint(self.n, size=count)
        k = where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k
