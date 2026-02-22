#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:08:09 2020

@author: dan
"""

import numpy as np
import pandas as pd

from py_tools import in_out, stats as st

def get_weighted_quantile_inner(df, var_list, weight_var, q, **kwargs):
    """Compute a single weighted quantile for each variable in *var_list*.

    Helper function used by :func:`collapse_quantile` when applied via
    ``groupby(...).apply()``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data for a single group.
    var_list : list of str
        Column names for which the quantile is computed.
    weight_var : str
        Column name of the weight variable.
    q : float
        Quantile level in [0, 1].
    **kwargs
        Additional keyword arguments forwarded to
        :func:`py_tools.stats.weighted_quantile`.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with one column per variable in *var_list*
        containing the weighted quantile value.
    """
    weights = df[weight_var].values
    
    data = {
        var : st.weighted_quantile(df[var].values, weights, [q], **kwargs)
        for var in var_list
        }
    
    return pd.DataFrame(data=data)

def collapse_quantile(df, by_list, weight_var=None, var_list=None, q=0.5, **kwargs):
    """Collapse a DataFrame to (weighted) quantiles within groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    by_list : list of str
        Columns to group by.
    weight_var : str or None, optional
        Column name of the weight variable. If ``None``, unweighted median
        is computed. Defaults to ``None``.
    var_list : list of str or None, optional
        Columns to summarise. If ``None``, all columns not in *by_list* or
        *weight_var* are used.
    q : float, optional
        Quantile level in [0, 1]. Defaults to ``0.5`` (median).
    **kwargs
        Additional keyword arguments forwarded to
        :func:`get_weighted_quantile_inner`.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by *by_list* containing the quantile of each
        variable in *var_list* within each group.
    """
    if var_list is None:
        var_list = [var for var in df.columns if var not in (by_list + [weight_var])]
    
    if weight_var is None:
        return df.groupby(by_list)[var_list].median()
    
    df_out = df.groupby(by_list).apply(get_weighted_quantile_inner, var_list, weight_var, q, **kwargs)

    col_name = 'level_{}'.format(len(by_list))
    df_out = df_out.reset_index().drop(columns=col_name).set_index(by_list)

    return df_out


def concat(collapser_list, check=True):
    """Concatenate a list of :class:`Collapser` objects along the observation axis.

    All ``Collapser`` objects in *collapser_list* are stacked row-wise and
    then re-collapsed using the same ``by_list`` as the first element.

    Parameters
    ----------
    collapser_list : list of Collapser
        Collapser objects to combine. All must share the same ``var_list``,
        ``weight_var``, and ``by_list`` when *check* is ``True``.
    check : bool, optional
        If ``True`` (default), assert that all Collapser objects are
        compatible before concatenating.

    Returns
    -------
    Collapser
        A new :class:`Collapser` representing the combined, re-collapsed data.

    Raises
    ------
    AssertionError
        If *check* is ``True`` and the Collapser objects have incompatible
        ``var_list``, ``weight_var``, or ``by_list``.
    """
    col0 = collapser_list[0]
    var_list = col0.var_list.copy()
    weight_var = col0.weight_var
    by_list = col0.by_list.copy()
    
    if check:
        for col in collapser_list[1:]:
            assert col.var_list == var_list
            assert col.weight_var == weight_var
            assert col.by_list == by_list
    
    dfc = pd.concat([col.dfc for col in collapser_list], axis=0)
    
    col = Collapser(dfc=dfc, var_list=var_list, weight_var=weight_var, 
                     by_list=by_list)
    col.collapse(by_list, inplace=True)
    return col

def load_collapser(filename, add_suffix=True, by_list=None, weight_var=None):
    """Load a :class:`Collapser` object from disk.

    Parameters
    ----------
    filename : str
        Base file path (without suffix) used when saving the Collapser.
    add_suffix : bool, optional
        If ``True`` (default), the same suffix that was appended during
        :meth:`Collapser.save` is reconstructed and appended to *filename*.
    by_list : list of str or None, optional
        Group-by column names (used to reconstruct the suffix when
        *add_suffix* is ``True``). Defaults to ``[]``.
    weight_var : str or None, optional
        Weight variable name (used to reconstruct the suffix when
        *add_suffix* is ``True``).

    Returns
    -------
    Collapser
        The loaded :class:`Collapser` object.
    """
    if by_list is None: by_list = []
        
    col = Collapser(by_list=by_list, weight_var=weight_var)
    col.load(filename, add_suffix=add_suffix)
    return col

def create_suffix(by_list, weight_var):
    """Build the file-name suffix used by :class:`Collapser` save/load methods.

    Parameters
    ----------
    by_list : list of str
        Group-by variable names.
    weight_var : str
        Weight variable name.

    Returns
    -------
    str
        Underscore-joined suffix string, e.g. ``'_year_state_wtd_by_pop'``.
    """
    return '_'.join([''] + by_list + ['wtd_by', weight_var])

def collapse(df, by_list, var_list=None, weight_var=None, weight_suffix=False):
    """Collapse a DataFrame to weighted means within groups.

    Convenience wrapper around :class:`Collapser` that creates a Collapser,
    collapses it, and returns the resulting DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    by_list : list of str
        Columns to group by.
    var_list : list of str or None, optional
        Columns to aggregate. Defaults to ``[]``, which causes
        :class:`Collapser` to use all non-group columns automatically.
    weight_var : str or None, optional
        Column name of the weight variable.
    weight_suffix : bool, optional
        If ``True``, append ``'_<weight_var>'`` to output column names.
        Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Collapsed DataFrame indexed by *by_list*.
    """
    if var_list is None: var_list = []
    
    coll = Collapser(df, var_list=var_list, by_list=by_list, 
                     weight_var=weight_var)
    
    return coll.get_data(weight_suffix=weight_suffix)

def collapse_multiweight(df, weight_dict, by_list=None):
    """Collapse variables to weighted means using variable-specific weights.

    Each variable in *weight_dict* is collapsed using its own weight variable
    and the results are concatenated column-wise.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    weight_dict : dict mapping str to str
        Mapping from variable name to its corresponding weight variable name.
    by_list : list of str or None, optional
        Columns to group by. Defaults to ``[]`` (global collapse).

    Returns
    -------
    pandas.DataFrame
        Collapsed DataFrame with one column per variable in *weight_dict*,
        indexed by *by_list*.
    """
    if by_list is None: by_list = []
    
    return pd.concat(
        [collapse(df, by_list, var_list=[var], weight_var=weight_var)
         for var, weight_var in weight_dict.items()],
        axis=1)

def collapse_multiquantile(df, by_list, q_list, weight_var=None, var_list=None, **kwargs):
    """Collapse variables to multiple weighted quantiles within groups.

    For each quantile level in *q_list*, :func:`collapse_quantile` is called
    and the resulting columns are renamed with a ``'_p<q*100>'`` suffix before
    being concatenated column-wise.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    by_list : list of str
        Columns to group by.
    q_list : list of float
        Quantile levels in [0, 1] to compute.
    weight_var : str or None, optional
        Column name of the weight variable.
    var_list : list of str or None, optional
        Columns to summarise. If ``None``, all non-group, non-weight columns
        are used.
    **kwargs
        Additional keyword arguments forwarded to :func:`collapse_quantile`.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by *by_list* with columns
        ``'<var>_p<q*100>'`` for each variable and quantile combination.
    """
    dfout_list = []
    for q in q_list:
        dfout = collapse_quantile(df, by_list, 
                                  weight_var= weight_var,
                                  var_list=var_list, q=q, **kwargs)
        
        dfout.columns = [col + '_p{}'.format(q*100) for col in dfout.columns]
        dfout_list.append(dfout)
    return pd.concat(dfout_list, axis=1)
        
def collapse_multiweight_multiquantile(df, weight_dict, q_list, by_list= []):
    """Collapse variables to multiple quantiles using variable-specific weights.

    Combines :func:`collapse_multiquantile` and :func:`collapse_multiweight`:
    each variable is collapsed at all quantile levels in *q_list* using its
    own weight variable, and the results are concatenated column-wise.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    weight_dict : dict mapping str to str
        Mapping from variable name to its corresponding weight variable name.
    q_list : list of float
        Quantile levels in [0, 1] to compute.
    by_list : list of str, optional
        Columns to group by. Defaults to ``[]`` (global collapse).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by *by_list* with columns
        ``'<var>_p<q*100>'`` for each variable and quantile combination.
    """
    return pd.concat(
        [collapse_multiquantile(df, by_list, q_list = q_list, weight_var = weight_var, var_list=[var])
         for var, weight_var in weight_dict.items()],
        axis=1)
    


class Collapser:
    """Class for flexibly collapsing datasets to weighted means.

    Stores numerator (weighted value) and denominator (total weight) columns
    internally so that multiple collapse operations can be chained without
    recomputing weights from scratch. The underlying DataFrame ``dfc`` has
    ``'<var>_num'`` and ``'<var>_denom'`` columns for each variable.

    Parameters
    ----------
    df : pandas.DataFrame or None, optional
        Raw input data. Either *df* or *dfc* must be provided, not both.
    var_list : list of str, optional
        Variables to aggregate. Defaults to ``[]``.
    weight_var : str or None, optional
        Column name of the weight variable. If ``None``, equal weights are
        assumed.
    by_list : list of str, optional
        Columns to group by. Defaults to ``[]``.
    dfc : pandas.DataFrame or None, optional
        Pre-constructed numerator/denominator DataFrame (used internally when
        creating a Collapser from an existing one, e.g. in :func:`concat`).
    **kwargs
        Additional keyword arguments forwarded to :meth:`set_data`.

    Attributes
    ----------
    dfc : pandas.DataFrame
        Internal DataFrame with ``'<var>_num'`` and ``'<var>_denom'`` columns.
    var_list : list of str
        Variables being aggregated.
    weight_var : str or None
        Weight variable name.
    by_list : list of str
        Current group-by variable names.
    """
    
    def __init__(self, df=None, var_list=None, weight_var=None, by_list=None, 
                 dfc=None, **kwargs):
        """Initialise a Collapser.

        Parameters
        ----------
        df : pandas.DataFrame or None, optional
            Raw input data. Mutually exclusive with *dfc*.
        var_list : list of str or None, optional
            Variables to aggregate. Defaults to ``[]``.
        weight_var : str or None, optional
            Weight variable column name.
        by_list : list of str or None, optional
            Group-by columns. Defaults to ``[]``.
        dfc : pandas.DataFrame or None, optional
            Pre-built numerator/denominator DataFrame.
        **kwargs
            Forwarded to :meth:`set_data`.

        Raises
        ------
        AssertionError
            If both *df* and *dfc* are provided.
        """

        if by_list is None: by_list = []
        if var_list is None: var_list = []
    
        assert (df is None) or (dfc is None)
        
        if df is not None:
            self.set_data(df, var_list, weight_var, by_list=by_list, inplace=True, 
                     **kwargs)
        else:
            self.dfc = dfc
            self.var_list = var_list
            self.weight_var = weight_var
            self.by_list = by_list
    
    def set_data(self, df, var_list, weight_var, by_list, collapse=True,
            inplace=True, scale=False):
        """Populate the Collapser from a raw DataFrame.

        Computes ``'<var>_num'`` (weighted value) and ``'<var>_denom'``
        (observation weight) columns for every variable in *var_list* and
        stores them in :attr:`dfc`.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw input data.
        var_list : list of str
            Variables to aggregate.
        weight_var : str or None
            Weight variable column name. If ``None``, unit weights are used.
        by_list : list of str
            Group-by columns.
        collapse : bool, optional
            If ``True`` (default), immediately collapse to *by_list* after
            building the numerator/denominator columns.
        inplace : bool, optional
            If ``True`` (default), update :attr:`dfc` in place; otherwise
            return a new :class:`Collapser`.
        scale : bool, optional
            If ``True``, normalise weights to have mean 1 before weighting.
            Defaults to ``False``.

        Returns
        -------
        Collapser or None
            If *collapse* is ``True`` and *inplace* is ``False``, returns a
            new :class:`Collapser`; otherwise returns ``None``.
        """
        copy_list = [var for var in by_list if var not in df.index.names]
        if copy_list:
            self.dfc = df[copy_list].copy()
        else:
            self.dfc = pd.DataFrame(index=df.index.copy())
        
        self.var_list = var_list
        self.weight_var = weight_var
        self.by_list = by_list
        
        if not self.var_list:
            self.var_list = [var for var in df.columns if var not in self.by_list]
            
        if self.weight_var is None:
            weight = np.ones((len(df), 1))
        else:
            weight = df[self.weight_var].values[:, np.newaxis]
            
        if scale:
            weight = weight / np.mean(weight)
            
        # Old way, caused fragmentation errors
        
        # for var in self.var_list:
            
        #     self.dfc[var + '_num'] = df[var] * weight
        #     self.dfc[var + '_denom'] = pd.notnull(df[var]).astype(np.int) * weight
            
        df_num = (df[self.var_list] * weight)
        df_denom = pd.notnull(df[self.var_list]) * weight
        
        df_num = df_num.rename({var : var + '_num' for var in df_num.columns}, axis=1)
        df_denom = df_denom.rename({var : var + '_denom' for var in df_denom.columns}, axis=1)
        
        self.dfc = pd.concat([self.dfc, df_num, df_denom], axis=1)
            
        if collapse:
            return self.collapse(self.by_list, inplace=inplace)
            
    def get_data(self, weight_suffix=False, include_denom=False):
        """Return the collapsed weighted-mean DataFrame.

        Divides numerator columns by denominator columns to produce weighted
        means, then optionally renames columns and appends denominator columns.

        Parameters
        ----------
        weight_suffix : bool, optional
            If ``True``, append ``'_<weight_var>'`` to each output column
            name. Defaults to ``False``.
        include_denom : bool, optional
            If ``True``, append the denominator (total weight) columns to the
            output. Defaults to ``False``.

        Returns
        -------
        pandas.DataFrame
            DataFrame of weighted means (and optionally denominators).
        """
        df_num, df_denom = self.get_numerators_and_denominators()
        
        df_out = df_num / df_denom
        if weight_suffix:
            df_out = df_out.rename({var : var + '_' + self.weight_var for var in self.var_list}, axis=1)
            
        if include_denom:
            df_out = pd.concat([df_out, self.dfc[[var + '_denom' for var in self.var_list]]], axis=1)
            
        return df_out
    
    def get_numerators_and_denominators(self):
        """Extract numerator and denominator DataFrames from :attr:`dfc`.

        Returns
        -------
        df_num : pandas.DataFrame
            Weighted-value columns (``'<var>_num'`` renamed to ``'<var>'``).
        df_denom : pandas.DataFrame
            Total-weight columns (``'<var>_denom'`` renamed to ``'<var>'``).
        """
        df_num = self.dfc[[var + '_num' for var in self.var_list]].rename({var + '_num' : var for var in self.var_list}, axis=1)
        df_denom = self.dfc[[var + '_denom' for var in self.var_list]].rename({var + '_denom' : var for var in self.var_list}, axis=1)
        
        return df_num, df_denom
            
    def collapse(self, by_list=None, inplace=False, method='mean'):
        """Collapse the internal DataFrame to *by_list* groups.

        Aggregates the numerator/denominator columns by summing within each
        group defined by *by_list*.

        Parameters
        ----------
        by_list : list of str or None, optional
            Columns to group by. An empty list (default) produces a single-row
            global collapse.
        inplace : bool, optional
            If ``True``, update :attr:`dfc` and :attr:`by_list` in place and
            return ``None``. If ``False`` (default), return a new
            :class:`Collapser`.
        method : {'mean'}, optional
            Aggregation method. Currently only ``'mean'`` is supported.

        Returns
        -------
        Collapser or None
            A new :class:`Collapser` when *inplace* is ``False``; ``None``
            when *inplace* is ``True``.

        Raises
        ------
        Exception
            If ``method='median'`` is requested (not yet implemented).
        """
        if by_list is None: by_list = []
        
        singleton = (not by_list)

        if method == 'mean':
            if singleton:
                dfc_new = pd.DataFrame([self.dfc.sum()])
            else:
                dfc_new = self.dfc.groupby(by_list).sum()
        elif method == 'median':
            raise Exception
            # dfc_new = dfc_old.groupby(by_list).agg(st.weighted_quantile, )
            
        if inplace:
            self.dfc = dfc_new
            self.by_list = by_list
            return None
        else:
            return Collapser(dfc=dfc_new, var_list=self.var_list.copy(), 
                             weight_var=self.weight_var, by_list=by_list)
        
    def resample(self, by_list, time_var, freq, inplace=False):
        """Resample the internal DataFrame over a time frequency within groups.

        Groups by *by_list*, then resamples the *time_var* level at *freq*.

        Parameters
        ----------
        by_list : list of str
            Non-time columns to group by.
        time_var : str
            Name of the time-level in the index to resample.
        freq : str
            Pandas offset alias (e.g. ``'Q'``, ``'A'``).
        inplace : bool, optional
            If ``True``, update :attr:`dfc` and :attr:`by_list` in place and
            return ``None``. If ``False`` (default), return a new
            :class:`Collapser`.

        Returns
        -------
        Collapser or None
            New :class:`Collapser` if *inplace* is ``False``; ``None``
            otherwise.
        """
        dfc_new = self.dfc.groupby(by_list).resample(freq, level=time_var).sum()
        by_list_new = list(dfc_new.index.names)
        
        if inplace:
            self.dfc = dfc_new
            self.by_list = by_list_new
            return None
        else:
            return Collapser(dfc=dfc_new, var_list=self.var_list.copy(), 
                             weight_var=self.weight_var, by_list=by_list_new)
        
    def loc(self, sliced, copy=False):
        """Return a row-sliced view of this Collapser.

        Parameters
        ----------
        sliced : label, slice, boolean array, or callable
            Selection passed directly to ``DataFrame.loc``.
        copy : bool, optional
            If ``True``, make a copy of the sliced data so subsequent
            modifications do not affect the original. Defaults to ``False``.

        Returns
        -------
        Collapser
            New :class:`Collapser` with the sliced internal DataFrame and the
            same ``var_list``, ``weight_var``, and ``by_list``.
        """
        dfc_sliced = self.dfc.loc[sliced, :]
        if copy:
            dfc_new = dfc_sliced.copy()
        else:
            dfc_new = dfc_sliced
            
        return Collapser(dfc=dfc_new, var_list=self.var_list.copy(),
                         weight_var=self.weight_var, by_list=self.by_list.copy())
    
    def get_weight(self, var):
        """Return the total weight (denominator) array for a variable.

        Parameters
        ----------
        var : str
            Variable name (must be in :attr:`var_list`).

        Returns
        -------
        numpy.ndarray
            1-D array of total weights for *var* from the ``'<var>_denom'``
            column of :attr:`dfc`.
        """
        return self.dfc[var + '_denom'].values
        
    def save(self, filename, add_suffix=True, fmt='parquet'):
        """Save the Collapser to disk.

        Persists :attr:`dfc` as a Parquet or pickle file and saves
        ``var_list``, ``weight_var``, and ``by_list`` as separate pickle
        files so the object can be fully reconstructed by :meth:`load`.

        Parameters
        ----------
        filename : str
            Base file path (without suffix or extension).
        add_suffix : bool, optional
            If ``True`` (default), append a descriptive suffix constructed
            from :attr:`by_list` and :attr:`weight_var` to *filename*.
        fmt : {'parquet', 'pickle'}, optional
            Storage format for :attr:`dfc`. Defaults to ``'parquet'``.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If *fmt* is not ``'parquet'`` or ``'pickle'``.
        """
        if add_suffix:
            suffix = create_suffix(self.by_list, self.weight_var)
        else:
            suffix = ''
            
        fullname = filename + suffix
        
        if fmt == 'parquet':
            self.dfc.to_parquet(fullname + '_data.parquet')
        elif fmt == 'pickle':
            self.dfc.to_pickle(fullname + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            in_out.save_pickle(getattr(self, item), fullname + '_' + item + '.pkl')

        return None
    
    def load(self, filename, add_suffix=True, fmt='parquet'):
        """Load Collapser data from disk.

        Reads :attr:`dfc` and the ``var_list``, ``weight_var``, and
        ``by_list`` metadata previously written by :meth:`save`.

        Parameters
        ----------
        filename : str
            Base file path (without suffix or extension).
        add_suffix : bool, optional
            If ``True`` (default), reconstruct and append the suffix that
            was used during :meth:`save`.
        fmt : {'parquet', 'pickle'}, optional
            Storage format for :attr:`dfc`. Defaults to ``'parquet'``.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If *fmt* is not ``'parquet'`` or ``'pickle'``.
        """
        if add_suffix:
            suffix = create_suffix(self.by_list, self.weight_var)
        else:
            suffix = ''
            
        fullname = filename + suffix
        
        if fmt == 'parquet':
            self.dfc = pd.read_parquet(fullname + '_data.parquet')
        elif fmt == 'pickle':
            self.dfc = pd.read_pickle(fullname + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            setattr(self, item, in_out.load_pickle(fullname + '_' + item + '.pkl'))

        return None
    
    def rename(self, name_map):
        """Rename variables in-place throughout the Collapser.

        Updates :attr:`by_list`, :attr:`var_list`, :attr:`weight_var`,
        the index of :attr:`dfc`, and the ``'_num'`` / ``'_denom'`` columns
        of :attr:`dfc` according to *name_map*.

        Parameters
        ----------
        name_map : dict mapping str to str
            Mapping from old variable name to new variable name. Variables
            not present in the map are left unchanged.

        Returns
        -------
        None
        """
        self.dfc.index.names = [name_map.get(var, var) for var in self.dfc.index.names]
        self.by_list = [name_map.get(var, var) for var in self.by_list]
        self.var_list = [name_map.get(var, var) for var in self.var_list]
        self.weight_var = name_map.get(self.weight_var, self.weight_var)
        
        name_map_dfc = {}
        for suffix in ['num', 'denom']:
            name_map_dfc.update({
                key + '_' + suffix : val + '_' + suffix for key, val in name_map.items()
                })
        
        self.dfc = self.dfc.rename(columns=name_map_dfc)
        
        return None
