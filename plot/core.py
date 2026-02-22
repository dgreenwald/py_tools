import matplotlib
import os

if os.environ.get("USE_MATPLOTLIB_AGG", 0):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.style as plt_style
import numpy as np
import pandas as pd
from scipy.stats import norm

from py_tools import data as dt, stats

pd.plotting.register_matplotlib_converters()


def set_fontsizes(ax, fontsize):
    """Set the font size for all text elements on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis whose text elements will be resized.
    fontsize : int or float
        The font size to apply to the title, axis labels, and tick labels.
    """
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)


def save_hist(vals, path, **kwargs):
    """Compute a histogram and save it as a pickle file.

    Parameters
    ----------
    vals : array-like
        Input data for the histogram.
    path : str
        Destination file path for the pickled histogram.
    **kwargs
        Additional keyword arguments forwarded to ``numpy.histogram``.
    """
    h = np.histogram(vals, **kwargs)
    dt.to_pickle(h, path)


def save_hist_npy(vals, base_path, **kwargs):
    """Compute a histogram and save the counts and bin edges as .npy files.

    Parameters
    ----------
    vals : array-like
        Input data for the histogram.
    base_path : str
        Path prefix used to derive output file names.  The histogram counts
        are saved to ``base_path + 'hist.npy'`` and the bin edges to
        ``base_path + 'bin_edges.npy'``.
    **kwargs
        Additional keyword arguments forwarded to ``numpy.histogram``.
    """
    hist, bin_edges = np.histogram(vals, **kwargs)
    np.save(base_path + "hist.npy", hist)
    np.save(base_path + "bin_edges.npy", bin_edges)


def load_hist_npy(base_path):
    """Load histogram counts and bin edges previously saved with :func:`save_hist_npy`.

    Parameters
    ----------
    base_path : str
        Path prefix used when the files were saved.  Reads
        ``base_path + 'hist.npy'`` and ``base_path + 'bin_edges.npy'``.

    Returns
    -------
    hist : numpy.ndarray
        Array of histogram bin counts.
    bin_edges : numpy.ndarray
        Array of bin edge values (length ``len(hist) + 1``).
    """
    hist = np.load(base_path + "hist.npy")
    bin_edges = np.load(base_path + "bin_edges.npy")

    return hist, bin_edges


def two_axis(
    df_in,
    var1,
    var2,
    filepath=None,
    loc1="upper left",
    loc2="upper right",
    loc_single=None,
    legend_font=10,
    label_font=12,
    normalize=False,
    color1=None,
    color2=None,
    colors=None,
    flip1=False,
    flip2=False,
    legend=True,
    single_legend=False,
    print_legend_axis=True,
    labels=None,
    leglabels=None,
    drop=True,
    kwargs1=None,
    kwargs2=None,
    format_dates=False,
    title=None,
    figsize=None,
    style=None,
    savefig_kwargs=None,
    axvline=None,
    axvline_kwargs=None,
):
    """Plot two time series on separate left and right y-axes.

    Parameters
    ----------
    df_in : pandas.DataFrame
        DataFrame containing both series.
    var1 : str
        Column name for the left-axis series.
    var2 : str
        Column name for the right-axis series.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    loc1 : str, optional
        Legend location for the left-axis legend.  Default is ``'upper left'``.
    loc2 : str, optional
        Legend location for the right-axis legend.  Default is ``'upper right'``.
    loc_single : str, optional
        Legend location when ``single_legend=True``.
    legend_font : int or float, optional
        Font size for legend text.  Default is 10.
    label_font : int or float, optional
        Font size for axis labels and tick labels.  Default is 12.
    normalize : bool, optional
        If ``True``, rescale both axes so that their normalized ranges match,
        making visual comparison of fluctuations easier.  Default is ``False``.
    color1 : str, optional
        Color for *var1*.  Falls back to ``colors[var1]`` then ``'#1f77b4'``.
    color2 : str, optional
        Color for *var2*.  Falls back to ``colors[var2]`` then ``'#ff7f0e'``.
    colors : dict, optional
        Mapping from variable name to color string, used as defaults when
        *color1* or *color2* are not provided.
    flip1 : bool, optional
        If ``True``, plot ``-var1`` and prefix the legend label with
        ``'(-1) x '``.  Default is ``False``.
    flip2 : bool, optional
        If ``True``, plot ``-var2`` and prefix the legend label with
        ``'(-1) x '``.  Default is ``False``.
    legend : bool, optional
        Whether to draw any legend.  Default is ``True``.
    single_legend : bool, optional
        If ``True``, combine both series into a single legend on *ax1*.
        Default is ``False``.
    print_legend_axis : bool, optional
        If ``True``, append ``' (left)'`` / ``' (right)'`` to legend labels.
        Default is ``True``.
    labels : dict, optional
        Mapping from variable name to display label used on the axis and in the
        legend.
    leglabels : dict, optional
        Mapping from variable name to legend-only label (overrides *labels* for
        the legend).
    drop : bool, optional
        If ``True``, drop rows where either variable is ``NaN`` before
        plotting.  Default is ``True``.
    kwargs1 : dict, optional
        Extra keyword arguments forwarded to the ``ax1.plot`` call for *var1*.
    kwargs2 : dict, optional
        Extra keyword arguments forwarded to the ``ax2.plot`` call for *var2*.
    format_dates : bool, optional
        If ``True``, call ``fig.autofmt_xdate()`` to rotate date labels.
        Default is ``False``.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        ``(width, height)`` in inches passed to ``plt.subplots``.
    style : str, optional
        Matplotlib style sheet name to activate before plotting.
    savefig_kwargs : dict, optional
        Extra keyword arguments forwarded to ``plt.savefig``.
    axvline : scalar, optional
        x-coordinate at which to draw a vertical line on *ax1*.
    axvline_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax1.axvline``.

    Returns
    -------
    None
    """

    if colors is None:
        colors = {}
    if labels is None:
        labels = {}
    if leglabels is None:
        leglabels = {}
    if kwargs1 is None:
        kwargs1 = {}
    if kwargs2 is None:
        kwargs2 = {}
    if savefig_kwargs is None:
        savefig_kwargs = {}
    if axvline_kwargs is None:
        axvline_kwargs = {}

    if style is not None:
        plt_style.use(style)

    kwargs1_copy = kwargs1.copy()
    kwargs1 = {
        "linewidth": 2,
    }
    kwargs1.update(kwargs1_copy)

    kwargs2_copy = kwargs2.copy()
    kwargs2 = {
        "linewidth": 2,
        "marker": "o",
        "markevery": 4,
        "fillstyle": "none",
        "markersize": 5,
        "mew": 1.5,
    }
    kwargs2.update(kwargs2_copy)

    matplotlib.rcParams.update({"font.size": label_font})

    df = df_in[[var1, var2]].copy()
    if drop:
        df = df.dropna()

    for these_kwargs in [kwargs1, kwargs2]:
        if these_kwargs.get("marker", None) is not None:
            if these_kwargs.get("markevery", None) is None:
                these_kwargs["markevery"] = max(1, len(df) // 20)

    if color1 is None:
        color1 = colors.get(var1, "#1f77b4")
    if color2 is None:
        color2 = colors.get(var2, "#ff7f0e")

    fig, ax1 = plt.subplots(figsize=figsize)

    label1 = labels.get(var1, var1)
    label2 = labels.get(var2, var2)

    leglabel1 = leglabels.get(var1, label1)
    leglabel2 = leglabels.get(var2, label2)
    if print_legend_axis:
        leglabel1 = leglabel1 + " (left)"
        leglabel2 = leglabel2 + " (right)"

    if flip1:
        line1 = ax1.plot(
            df.index, -df[var1], label=("(-1) x " + leglabel1), color=color1, **kwargs1
        )
    else:
        line1 = ax1.plot(df.index, df[var1], label=leglabel1, color=color1, **kwargs1)

    ax2 = ax1.twinx()
    if flip2:
        line2 = ax2.plot(
            df.index, -df[var2], label=("(-1) x " + leglabel2), color=color2, **kwargs2
        )
    else:
        line2 = ax2.plot(df.index, df[var2], label=leglabel2, color=color2, **kwargs2)

    if legend:
        if single_legend:
            these_lines = line1 + line2
            these_labels = [line.get_label() for line in these_lines]
            ax1.legend(these_lines, these_labels, loc=loc_single, fontsize=legend_font)
        else:
            ax1.legend(loc=loc1, fontsize=legend_font)
            ax2.legend(loc=loc2, fontsize=legend_font)

    ax1.set_ylabel(label1, color=color1, fontsize=label_font)
    for tl in ax1.get_yticklabels():
        tl.set_color(color1)

    ax2.set_ylabel(label2, color=color2, fontsize=label_font)
    for tl in ax2.get_yticklabels():
        tl.set_color(color2)

    if normalize:
        ax1_ylim = ax1.get_ylim()
        ax2_ylim = ax2.get_ylim()

        ax1_ylim_norm = (np.array(ax1_ylim) - df[var1].mean()) / df[var1].std()
        ax2_ylim_norm = (np.array(ax2_ylim) - df[var2].mean()) / df[var2].std()

        ylim_norm = np.array(
            [
                np.minimum(ax1_ylim_norm[0], ax2_ylim_norm[0]),
                np.maximum(ax1_ylim_norm[1], ax2_ylim_norm[1]),
            ]
        )

        ax1_ylim_new = df[var1].std() * ylim_norm + df[var1].mean()
        ax2_ylim_new = df[var2].std() * ylim_norm + df[var2].mean()

        ax1.set_ylim(ax1_ylim_new)
        ax2.set_ylim(ax2_ylim_new)

    if axvline is not None:
        ax1.axvline(axvline, **axvline_kwargs)

    plt.xlim(df.index[[0, -1]])

    if title is not None:
        plt.title(title)

    if format_dates:
        fig.autofmt_xdate()

    if filepath is not None:
        plt.tight_layout()
        plt.savefig(filepath, **savefig_kwargs)
    else:
        plt.show()

    plt.close(fig)

    return None


def normalized(df, var_list, filepath=None, invert_list=None):
    """Plot multiple series after standardizing each to zero mean and unit variance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose index will be used as the x-axis.
    var_list : list of str
        Column names to plot.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    invert_list : list of bool, optional
        A bool for each variable in *var_list*.  When ``True`` the
        standardized series is multiplied by ``-1`` and the legend entry is
        prefixed with ``'(-1) x '``.  Defaults to all ``False``.

    Returns
    -------
    None
    """
    if invert_list is None:
        invert_list = len(var_list) * [False]

    fig = plt.figure()

    for this_var, invert in zip(var_list, invert_list):
        x = df[this_var].values.copy()
        x -= np.mean(x)
        x /= np.std(x)

        if invert:
            x *= -1
            invert_str = "(-1) x "
        else:
            invert_str = ""

        plt.plot(df.index, x, label=invert_str + this_var)

    plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    return None


def hist(
    df_in,
    var,
    label=None,
    xlabel=None,
    ylabel=None,
    wvar=None,
    bins=None,
    xlim=None,
    ylim=None,
    filepath=None,
    legend_font=10,
    label_font=12,
    copy_path=None,
    x_vertline=None,
    vertline_kwargs=None,
    **kwargs,
):
    """Plot a weighted, normalized histogram for a single variable.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame containing the variable to plot.
    var : str
        Column name of the variable to histogram.
    label : str, optional
        Legend label for the histogram.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    wvar : str, optional
        Column name of observation weights.  Uses equal weights when
        ``None``.
    bins : int or sequence, optional
        Bin specification forwarded to ``plt.hist``.
    xlim : tuple of float, optional
        ``(left, right)`` x-axis limits.
    ylim : tuple of float, optional
        ``(bottom, top)`` y-axis limits.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    legend_font : int or float, optional
        Font size for legend text.  Default is 10.
    label_font : int or float, optional
        Font size for axis labels.  Default is 12.
    copy_path : str, optional
        If provided, also save the histogram data as ``.npy`` files using
        :func:`save_hist_npy` with this path prefix.
    x_vertline : float, optional
        x-coordinate at which to draw a vertical line.
    vertline_kwargs : dict, optional
        Extra keyword arguments forwarded to ``plt.axvline``.
    **kwargs
        Additional keyword arguments forwarded to ``plt.hist``.

    Returns
    -------
    bool
        ``True`` if the plot was produced, ``False`` if *var* was not found in
        the cleaned DataFrame or the DataFrame was empty after cleaning.
    """

    if vertline_kwargs is None:
        vertline_kwargs = {}

    df = dt.clean(df_in, [var, wvar])

    if var not in df or len(df) == 0:
        return False

    if wvar is not None:
        w = df[wvar].values
    else:
        w = np.ones(len(df))

    # Normalize
    kwargs["density"] = True

    # TODO: could use kwargs for some of these
    fig = plt.figure()
    matplotlib.rcParams.update({"font.size": label_font})
    plt.hist(
        df[var].values,
        bins=bins,
        alpha=0.5,
        edgecolor="black",
        weights=w,
        label=label,
        **kwargs,
    )

    if x_vertline is not None:
        plt.axvline(x=x_vertline, **vertline_kwargs)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim((xlim))
    if ylim is not None:
        plt.ylim((ylim))
    if label is not None:
        plt.legend(fontsize=legend_font)

    if filepath is not None:
        plt.tight_layout()
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    if copy_path is not None:
        save_hist_npy(df[var].values, copy_path, density=True, bins=bins, weights=w)

    return True


def compute_hist(df, var, bins, wvar=None):
    """Compute a normalized weighted histogram and return it as a Series.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    var : str
        Column name of the variable to bin.
    bins : array-like
        Monotonically increasing sequence of bin edges.
    wvar : str, optional
        Column name of observation weights.  Uses equal weights (all ones)
        when ``None``.

    Returns
    -------
    pandas.Series
        Normalized bin weights (sums to 1) indexed by the left edge of each
        bin.
    """

    this_list = [var]
    if wvar is not None:
        this_list.append(wvar)

    _df = df[this_list].copy()
    if wvar is None:
        wvar = "_ONES_"
        _df[wvar] = 1.0

    _df["bin"] = pd.cut(_df[var], bins, labels=bins[:-1])
    hist = _df.groupby("bin", observed=False)[wvar].sum()
    hist /= np.sum(hist)

    return hist


def multi_hist(
    dfs,
    labels=None,
    xvar=None,
    xvars=None,
    bins=None,
    wvar=None,
    wvars=None,
    filepath=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    legend_font=10,
    label_font=12,
    copy_paths=None,
    colors=None,
    edgecolor="black",
    alpha=0.5,
    use_bar=False,
    kwarg_list=None,
    x_vertline=None,
    vertline_kwargs=None,
    topcode=False,
    bottomcode=False,
    density=True,
    **kwargs,
):
    """Plot overlapping histograms for multiple datasets or variables.

    Each element of *dfs* (or each variable in *xvars*) is drawn as a
    separate histogram on the same axes.

    Parameters
    ----------
    dfs : list of pandas.DataFrame or pandas.DataFrame
        One DataFrame per histogram group.  May be a single DataFrame
        (replicated for all groups via *xvars*).
    labels : list of str, optional
        Legend label for each histogram group.
    xvar : str, optional
        Fallback column name used for all groups when *xvars* is not
        provided.
    xvars : list of str, optional
        Column name to histogram for each group.
    bins : int or array-like, optional
        Bin specification forwarded to ``plt.hist`` or ``plt.bar``.
    wvar : str, optional
        Fallback weight column used for all groups when *wvars* is not
        provided.  Defaults to equal counts when ``None``.
    wvars : list of str, optional
        Weight column name for each group.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xlim : tuple of float, optional
        ``(left, right)`` x-axis limits.  Defaults to the outer bin edges
        when *bins* is provided.
    ylim : tuple of float, optional
        ``(bottom, top)`` y-axis limits.
    legend_font : int or float, optional
        Font size for legend text.  Default is 10.
    label_font : int or float, optional
        Font size for axis labels.  Default is 12.
    copy_paths : list of str, optional
        If provided, save each histogram's data as a pickle using
        :func:`save_hist` with the corresponding path.
    colors : list of str, optional
        Bar/line color for each group.
    edgecolor : str, optional
        Edge color for histogram bars.  Default is ``'black'``.
    alpha : float, optional
        Transparency of histogram bars.  Default is 0.5.
    use_bar : bool, optional
        If ``True``, use ``plt.bar`` instead of ``plt.hist``.  Requires
        *bins* to be provided.  Default is ``False``.
    kwarg_list : list of dict, optional
        Per-group extra keyword arguments merged into the plot call.
    x_vertline : float, optional
        x-coordinate at which to draw a vertical line.
    vertline_kwargs : dict, optional
        Extra keyword arguments forwarded to ``plt.axvline``.
    topcode : bool, optional
        If ``True``, clip values at the last bin edge.  Default is ``False``.
    bottomcode : bool, optional
        If ``True``, clip values at the first bin edge.  Default is
        ``False``.
    density : bool, optional
        If ``True`` (default) and ``use_bar=False``, normalize histograms so
        that the area sums to 1.
    **kwargs
        Additional keyword arguments forwarded to every ``plt.hist`` call.

    Returns
    -------
    bool
        Always returns ``True``.
    """

    if vertline_kwargs is None:
        vertline_kwargs = {}

    def get_length(x):
        if isinstance(x, list):
            return len(x)
        else:
            return 0

    def to_list(x_in, n_plots, default=None):

        if x_in is None:
            x_in = default

        if isinstance(x_in, list):
            x = x_in.copy()
        else:
            x = [x_in]

        if (len(x) == 1) and (n_plots > 1):
            x = n_plots * x

        assert len(x) == n_plots

        return x

    # Number of independent plots
    n_plots = max([get_length(x) for x in [dfs, xvars, wvars]])
    ilist = list(range(n_plots))

    # Default: equal weights
    if wvar is None:
        wvar = "_count"

    # Bound at edges of bins
    if (xlim is None) and (bins is not None):
        xlim = bins[[0, -1]]

    # Normalize
    if density and (not use_bar):
        kwargs["density"] = True

    dfs = to_list(dfs, n_plots)
    labels = to_list(labels, n_plots)
    colors = to_list(colors, n_plots)

    xvars = to_list(xvars, n_plots, xvar)
    wvars = to_list(wvars, n_plots, wvar)
    kwarg_list = to_list(kwarg_list, n_plots, {})

    for ii in ilist:
        these_kwargs = kwargs.copy()
        these_kwargs.update(kwarg_list[ii])
        kwarg_list[ii] = these_kwargs

    _dfs = []

    for ii in ilist:
        xv = xvars[ii]
        wv = wvars[ii]

        df = dfs[ii][[xv]].copy()

        if wvars[ii] == "_count":
            df["_count"] = 1.0
        else:
            df[wv] = dfs[ii][wv]

        df = dt.clean(df, [xv, wv])

        if topcode:
            df[xv] = np.minimum(df[xv], bins[-1])

        if bottomcode:
            df[xv] = np.maximum(df[xv], bins[0])

        _dfs.append(df)

    fig = plt.figure()
    matplotlib.rcParams.update({"font.size": label_font})

    if use_bar:
        assert bins is not None

        # TODO:
        bins = np.array(bins)
        bin_width = bins[1:] - bins[:-1]

        for ii in ilist:
            hist = compute_hist(_dfs[ii], xvars[ii], bins, wvar=wvars[ii])

            plt.bar(
                np.array(hist.index),
                hist.values,
                width=bin_width,
                alpha=alpha,
                edgecolor=edgecolor,
                align="edge",
                label=str(labels[ii]),
                color=colors[ii],
                **kwarg_list[ii],
            )

    else:
        for ii in ilist:
            plt.hist(
                _dfs[ii][xvars[ii]].values,
                bins=bins,
                alpha=alpha,
                edgecolor=edgecolor,
                weights=_dfs[ii][wvars[ii]].copy(),
                label=str(labels[ii]),
                color=colors[ii],
                **kwarg_list[ii],
            )

    if x_vertline is not None:
        plt.axvline(x=x_vertline, **vertline_kwargs)

    plt.legend(fontsize=legend_font)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_font)

    if xlim is not None:
        plt.xlim((xlim))
    if ylim is not None:
        plt.ylim((ylim))

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    if copy_paths is not None:
        for ii in ilist:
            save_hist(
                _dfs[ii][xvars[ii]].values,
                copy_paths[ii],
                density=True,
                bins=bins,
                weights=_dfs[ii][wvars[ii]].values,
            )

    return True


def double_hist(
    df1,
    df2=None,
    label1=None,
    label2=None,
    var=None,
    var1=None,
    var2=None,
    bins=None,
    wvar=None,
    wvar1=None,
    wvar2=None,
    filepath=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    legend_font=10,
    label_font=12,
    copy_path1=None,
    copy_path2=None,
    color1=None,
    color2=None,
    edgecolor="black",
    alpha=0.5,
    use_bar=False,
    labels=None,
    kwargs1=None,
    kwargs2=None,
    x_vertline=None,
    vertline_kwargs=None,
    topcode=False,
    bottomcode=False,
    **kwargs,
):
    """Plot two overlapping histograms on the same axes.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame for the first histogram.
    df2 : pandas.DataFrame, optional
        DataFrame for the second histogram.  Defaults to *df1* when
        ``None``.
    label1 : str, optional
        Legend label for the first histogram.
    label2 : str, optional
        Legend label for the second histogram.
    var : str, optional
        Column name used for both histograms.  Mutually exclusive with
        *var1* and *var2*.
    var1 : str, optional
        Column name for the first histogram.
    var2 : str, optional
        Column name for the second histogram.
    bins : int or array-like, optional
        Bin specification forwarded to ``plt.hist`` or ``plt.bar``.
    wvar : str, optional
        Default weight column for both histograms.  Uses equal counts when
        ``None``.
    wvar1 : str, optional
        Weight column for the first histogram.  Falls back to *wvar*.
    wvar2 : str, optional
        Weight column for the second histogram.  Falls back to *wvar*.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xlim : tuple of float, optional
        ``(left, right)`` x-axis limits.  Defaults to the outer bin edges
        when *bins* is provided.
    ylim : tuple of float, optional
        ``(bottom, top)`` y-axis limits.
    legend_font : int or float, optional
        Font size for legend text.  Default is 10.
    label_font : int or float, optional
        Font size for axis labels.  Default is 12.
    copy_path1 : str, optional
        If provided, save the first histogram as a pickle using
        :func:`save_hist` with this path.
    copy_path2 : str, optional
        If provided, save the second histogram as a pickle using
        :func:`save_hist` with this path.
    color1 : str, optional
        Color for the first histogram bars.
    color2 : str, optional
        Color for the second histogram bars.
    edgecolor : str, optional
        Edge color for histogram bars.  Default is ``'black'``.
    alpha : float, optional
        Transparency of histogram bars.  Default is 0.5.
    use_bar : bool, optional
        If ``True``, use ``plt.bar`` instead of ``plt.hist``.  Requires
        *bins* to be provided.  Default is ``False``.
    labels : dict, optional
        Mapping from variable name to display label, used as fallback when
        *label1* or *label2* are not provided.
    kwargs1 : dict, optional
        Extra keyword arguments forwarded to the first histogram plot call.
    kwargs2 : dict, optional
        Extra keyword arguments forwarded to the second histogram plot call.
    x_vertline : float, optional
        x-coordinate at which to draw a vertical line.
    vertline_kwargs : dict, optional
        Extra keyword arguments forwarded to ``plt.axvline``.
    topcode : bool, optional
        If ``True``, clip values at the last bin edge.  Default is ``False``.
    bottomcode : bool, optional
        If ``True``, clip values at the first bin edge.  Default is
        ``False``.
    **kwargs
        Additional keyword arguments merged into both *kwargs1* and
        *kwargs2*.

    Returns
    -------
    bool
        ``True`` if the plot was produced, ``False`` if either variable was
        missing or the cleaned DataFrames were empty.
    """

    if labels is None:
        labels = {}
    if kwargs1 is None:
        kwargs1 = {}
    if kwargs2 is None:
        kwargs2 = {}
    if vertline_kwargs is None:
        vertline_kwargs = {}

    kwargs1.update(kwargs)
    kwargs2.update(kwargs)

    if df2 is None:
        df2 = df1

    if var is not None:
        assert var1 is None and var2 is None
        var1 = var
        var2 = var

    if wvar is None:
        wvar = "_count"

    if wvar1 is None:
        wvar1 = wvar

    if wvar2 is None:
        wvar2 = wvar

    if label1 is None:
        label1 = labels.get(var1, var1)

    if label2 is None:
        label2 = labels.get(var2, var2)

    # Normalize
    kwargs["density"] = True

    _df1 = df1[[var1]].copy()
    _df2 = df2[[var2]].copy()

    if wvar1 == "_count":
        _df1["_weight1"] = 1.0
    else:
        _df1["_weight1"] = df1[wvar1]

    if wvar2 == "_count":
        _df2["_weight2"] = 1.0
    else:
        _df2["_weight2"] = df2[wvar2]

    wvar1 = "_weight1"
    wvar2 = "_weight2"

    _df1 = dt.clean(_df1, [var1, wvar1])
    _df2 = dt.clean(_df2, [var2, wvar2])

    if topcode:
        assert bins is not None
        _df1[var1] = np.minimum(_df1[var1], bins[-1])
        _df2[var2] = np.minimum(_df2[var2], bins[-1])

    if bottomcode:
        assert bins is not None
        _df1[var1] = np.maximum(_df1[var1], bins[0])
        _df2[var2] = np.maximum(_df2[var2], bins[0])

    if var1 not in _df1 or var2 not in _df2:
        return False

    if len(_df1) == 0 or len(_df2) == 0:
        return False

    if wvar1 is not None:
        w1 = _df1[wvar1].values
    else:
        w1 = np.ones(len(_df1))

    if wvar2 is not None:
        w2 = _df2[wvar2].values
    else:
        w2 = np.ones(len(_df2))

    if (xlim is None) and (bins is not None):
        xlim = bins[[0, -1]]

    fig = plt.figure()
    matplotlib.rcParams.update({"font.size": label_font})

    if use_bar:
        assert bins is not None

        bin_width = bins[1] - bins[0]

        hist1 = compute_hist(_df1, var1, bins, wvar=wvar1)
        hist2 = compute_hist(_df2, var2, bins, wvar=wvar2)

        plt.bar(
            np.array(hist1.index),
            hist1.values,
            width=bin_width,
            alpha=alpha,
            edgecolor=edgecolor,
            align="edge",
            label=str(label1),
            color=color1,
            **kwargs1,
        )
        plt.bar(
            np.array(hist2.index),
            hist2.values,
            width=bin_width,
            alpha=alpha,
            edgecolor=edgecolor,
            align="edge",
            label=str(label2),
            color=color2,
            **kwargs2,
        )

    else:
        plt.hist(
            _df1[var1].values,
            bins=bins,
            alpha=alpha,
            edgecolor=edgecolor,
            weights=w1,
            label=str(label1),
            color=color1,
            **kwargs1,
        )
        plt.hist(
            _df2[var2].values,
            bins=bins,
            alpha=alpha,
            edgecolor=edgecolor,
            weights=w2,
            label=str(label2),
            color=color2,
            **kwargs2,
        )

    if x_vertline is not None:
        plt.axvline(x=x_vertline, **vertline_kwargs)

    plt.legend(fontsize=legend_font)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_font)

    if xlim is not None:
        plt.xlim((xlim))
    if ylim is not None:
        plt.ylim((ylim))

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    if copy_path1 is not None:
        save_hist(_df1[var1].values, copy_path1, density=True, bins=bins, weights=w1)
    if copy_path2 is not None:
        save_hist(_df2[var2].values, copy_path2, density=True, bins=bins, weights=w2)

    return True


def var_irfs(
    irfs,
    var_list,
    shock_list=None,
    titles=None,
    filepath=None,
    n_per_row=None,
    plot_scale=3,
):
    """Plot impulse response functions (IRFs) from a posterior simulation.

    Creates a grid of subplots, one per (variable, shock) pair, showing the
    posterior median and 16th/84th percentile bands.

    Parameters
    ----------
    irfs : numpy.ndarray, shape (Nsim, Nirf, Ny, Nshock)
        Array of simulated impulse responses.  The axes correspond to:
        posterior draw, horizon, response variable, and shock.
    var_list : list of str
        Names of the response variables (length ``Ny``).
    shock_list : list of str, optional
        Names of the shocks (length ``Nshock``).  Defaults to *var_list*.
    titles : dict, optional
        Mapping from variable/shock name to display title.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    n_per_row : int, optional
        Number of subplot columns.  Defaults to ``Nshock``.
    plot_scale : float, optional
        Inches per subplot side.  Default is 3.

    Returns
    -------
    None
    """

    if titles is None:
        titles = {}

    if shock_list is None:
        shock_list = var_list

    Nsim, Nirf, Ny, Nshock = irfs.shape

    center = np.median(irfs, axis=0)
    bands = np.percentile(irfs, [16, 84], axis=0)

    if n_per_row is None:
        n_per_row = Nshock

    n_rows = (((Ny * Nshock) - 1) // n_per_row) + 1

    fig = plt.figure()
    for iy in range(Ny):
        for ishock in range(Nshock):
            plt.subplot(n_rows, n_per_row, Ny * ishock + iy + 1)

            plt.plot(np.zeros(Nirf), color="gray", linestyle=":")
            plt.plot(center[:, iy, ishock], color="blue")
            plt.plot(bands[0, :, iy, ishock], color="black", linestyle="--")
            plt.plot(bands[1, :, iy, ishock], color="black", linestyle="--")

            plt.xlim((0, Nirf - 1))

            var_title = titles.get(var_list[iy], var_list[iy])
            shock_title = titles.get(shock_list[ishock], shock_list[ishock])
            plt.title("{0} to {1}".format(var_title, shock_title))

    if filepath is None:
        plt.show()
    else:
        fig.set_size_inches((plot_scale * n_per_row, plot_scale * n_rows))
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)

    return None


def plot_series(
    df_in,
    var_names,
    filepath=None,
    directory=None,
    filename=None,
    labels=None,
    linestyles=None,
    markers=None,
    colors=None,
    markevery=8,
    markersize=5,
    mew=2,
    fillstyle="none",
    fontsize=12,
    plot_type=None,
    ylabel=None,
    sample="outer",
    title=None,
    single_legend=True,
    vertline_ix=None,
    vertline_kwargs=None,
    linewidths=None,
    ylim=None,
    dpi=None,
):
    """Plot one or more time series from a DataFrame.

    Parameters
    ----------
    df_in : pandas.DataFrame
        DataFrame whose index is used as the x-axis.
    var_names : list of str
        Column names to plot.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    directory : str, optional
        Deprecated.  Raises an exception if provided.
    filename : str, optional
        Deprecated.  Raises an exception if provided.
    labels : dict, optional
        Mapping from column name to legend label.
    linestyles : dict, optional
        Mapping from column name to line style string.  Defaults to
        ``'-'`` for each variable.
    markers : dict, optional
        Mapping from column name to marker style string.
    colors : dict, optional
        Mapping from column name to color string.
    markevery : int, optional
        Plot a marker every this many data points.  Default is 8.
    markersize : float, optional
        Marker size in points.  Default is 5.
    mew : float, optional
        Marker edge width.  Default is 2.
    fillstyle : str, optional
        Marker fill style.  Default is ``'none'``.
    fontsize : int or float, optional
        Global font size for labels and legend.  Default is 12.
    plot_type : str, optional
        Deprecated.  Raises an exception if provided.
    ylabel : str, optional
        Label for the y-axis.
    sample : {'outer', 'inner'}, optional
        Row selection strategy.  ``'outer'`` keeps rows where any variable is
        non-null; ``'inner'`` keeps only rows where all variables are
        non-null.  Default is ``'outer'``.
    title : str, optional
        Figure title.
    single_legend : bool, optional
        If ``True``, always draw a legend even when only one variable is
        plotted.  Default is ``True``.
    vertline_ix : int, optional
        Integer index into ``df.index`` at which to draw a vertical line.
    vertline_kwargs : dict, optional
        Extra keyword arguments forwarded to ``plt.axvline``.
    linewidths : dict, optional
        Mapping from column name to line width.  Defaults to 2 for each
        variable.
    ylim : tuple of float, optional
        ``(bottom, top)`` y-axis limits.
    dpi : int or float, optional
        Resolution for the saved figure.
    """

    if labels is None:
        labels = {}
    if linestyles is None:
        linestyles = {}
    if markers is None:
        markers = {}
    if colors is None:
        colors = {}
    if vertline_kwargs is None:
        vertline_kwargs = {}
    if linewidths is None:
        linewidths = {}

    matplotlib.rcParams.update({"font.size": fontsize})

    if (directory is not None) or (filename is not None) or (plot_type is not None):
        raise ValueError(
            "Deprecated arguments `directory`, `filename`, and `plot_type` are no longer supported. "
            "Use `filepath=` instead."
        )

    fig = plt.figure()

    if sample == "outer":
        ix = np.any(pd.notnull(df_in[var_names]), axis=1)
    elif sample == "inner":
        ix = np.all(pd.notnull(df_in[var_names]), axis=1)
    else:
        raise ValueError("`sample` must be 'outer' or 'inner'.")

    df = df_in.loc[ix, var_names].copy()

    for var in var_names:
        label = labels.get(var, var)
        linestyle = linestyles.get(var, "-")
        color = colors.get(var, None)
        linewidth = linewidths.get(var, 2)

        marker = markers.get(var, None)

        plt.plot(
            df.index,
            df[var].values,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            mew=mew,
            color=color,
        )

    if vertline_ix is not None:
        plt.axvline(x=df.index[vertline_ix], **vertline_kwargs)

    if len(var_names) > 1 or single_legend:
        plt.legend(fontsize=fontsize)

    plt.xlim(df.index[[0, -1]])

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()

    plt.close(fig)


def projection(
    x,
    se,
    var_titles,
    shock_title,
    p=0.9,
    n_per_row=4,
    plot_size=3.0,
    out_dir=None,
    label=None,
    shock_name=None,
):
    """Plot impulse responses from a local projection estimation.

    Parameters
    ----------
    x : numpy.ndarray, shape (Ny, Nt)
        Estimated coefficients, one row per response variable and one column
        per horizon.
    se : numpy.ndarray, shape (Ny, Nt)
        Standard errors corresponding to *x*.
    var_titles : list of str
        Display names for the response variables (length ``Ny``).
    shock_title : str
        Display name of the shock, used in subplot titles and the output
        filename.
    p : float, optional
        Probability coverage of the confidence bands (e.g. ``0.9`` gives 90%
        bands).  Default is 0.9.
    n_per_row : int, optional
        Number of subplots per row.  Default is 4.
    plot_size : float, optional
        Inches per subplot side.  Default is 3.0.
    out_dir : str, optional
        Directory in which to save the figure.  Displays interactively when
        ``None``.
    label : str, optional
        Optional prefix added to the output filename.
    shock_name : str, optional
        Shorter name used in the output filename instead of *shock_title*.

    Returns
    -------
    None
    """

    Ny, Nt = x.shape
    n_rows = (Ny - 1) // n_per_row + 1  # number of rows needed

    if n_rows == 1:
        n_per_row = Ny

    # Get z-score for p-value
    z_star = -norm.ppf(0.5 * (1.0 - p))

    # Plot
    fig = plt.figure()
    for iy in range(Ny):
        plt.subplot(n_rows, n_per_row, iy + 1)

        plt.plot(np.arange(Nt), np.zeros(x[iy, :].shape), linestyle=":", color="gray")
        plt.plot(np.arange(Nt), x[iy, :], linestyle="-", color="blue")
        plt.plot(
            np.arange(Nt), x[iy, :] + z_star * se[iy, :], linestyle="--", color="black"
        )
        plt.plot(
            np.arange(Nt), x[iy, :] - z_star * se[iy, :], linestyle="--", color="black"
        )

        plt.title("{0} to {1}".format(var_titles[iy], shock_title))

        plt.xlim((0, Nt - 1))
        plt.xlabel("Quarters")

    fig.set_size_inches((plot_size * n_per_row, plot_size * n_rows))
    plt.tight_layout()

    if out_dir is None:
        plt.show()

    else:
        if label is None:
            prefix = ""
        else:
            prefix = label + "_"

        if shock_name is None:
            shock_name = shock_title

        plt.savefig("{0}/{1}{2}_projections.pdf".format(out_dir, prefix, shock_name))

    plt.close(fig)

    return None


def get_45_bounds(df, xvar, yvar, margin=0.05):
    """Compute symmetric axis bounds for a 45-degree line plot.

    Finds a common lower and upper bound that covers the data range of both
    *xvar* and *yvar*, with an optional fractional margin.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the two variables.
    xvar : str
        Column name of the x-axis variable.
    yvar : str
        Column name of the y-axis variable.
    margin : float, optional
        Fractional margin to add beyond the data range.  Default is 0.05
        (5% on each side).

    Returns
    -------
    plot_lb : float
        Lower axis bound.
    plot_ub : float
        Upper axis bound.
    """
    min_val = np.amin(df[[yvar, xvar]].values)
    max_val = np.amax(df[[yvar, xvar]].values)
    dist = max_val - min_val

    plot_lb = min_val - margin * dist
    plot_ub = max_val + margin * dist

    return plot_lb, plot_ub


def binscatter(
    df_in,
    yvars,
    xvar,
    wvar=None,
    fit_var=None,
    labels=None,
    n_bins=20,
    bins=None,
    filepath=None,
    xlim=None,
    ylim=None,
    plot_line=True,
    control=None,
    absorb=None,
    bin_scale=None,
    raw_scale=10.0,
    plot_raw_data=False,
    bin_kwargs=None,
    raw_kwargs=None,
    line_kwargs=None,
    legend_font=10,
    label_font=12,
    use_legend=True,
    median=False,
    restore_mean=False,
    title=None,
    include45=False,
    include0=False,
    **kwargs,
):
    """Create a binned scatter plot with optional OLS fit line.

    Bins *xvar* into *n_bins* equal-frequency bins and plots the
    (weighted) mean of each *yvar* within each bin.  Optionally overlays
    raw data points and a fitted regression line.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame.
    yvars : str or list of str
        Response variable(s) to plot on the y-axis.
    xvar : str
        Explanatory variable to bin on the x-axis.
    wvar : str, optional
        Column name of observation weights.
    fit_var : str, optional
        Pre-computed fitted-values column.  When ``None`` the fit is
        obtained by running an OLS regression of *yvar* on *xvar*.
    labels : dict, optional
        Mapping from column name to display label.
    n_bins : int, optional
        Number of bins.  Default is 20.
    bins : array-like, optional
        Explicit bin edges.  When provided, *n_bins* is ignored.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    xlim : tuple of float or 'default', optional
        x-axis limits.
    ylim : tuple of float or 'default', optional
        y-axis limits.
    plot_line : bool, optional
        If ``True`` (default), overlay an OLS fit line.
    control : list of str, optional
        Control variables to partial out before binning.
    absorb : list of str or list of list of str, optional
        Fixed-effect variable(s) to absorb (demean) before binning.
    bin_scale : float, optional
        Marker size for binned scatter points.  Defaults to 50 without raw
        data and 100 with raw data.
    raw_scale : float, optional
        Scale factor applied to weights when drawing raw data points.
        Default is 10.0.
    plot_raw_data : bool, optional
        If ``True``, also plot the raw (unbinned) data points.  Default is
        ``False``.
    bin_kwargs : dict, optional
        Extra keyword arguments forwarded to the binned scatter ``ax.scatter``
        call.
    raw_kwargs : dict, optional
        Extra keyword arguments forwarded to the raw data ``ax.scatter`` call.
    line_kwargs : dict, optional
        Extra keyword arguments forwarded to the fit-line ``ax.plot`` call.
    legend_font : int or float, optional
        Font size for legend text.  Default is 10.
    label_font : int or float, optional
        Font size for axis labels.  Default is 12.
    use_legend : bool, optional
        If ``True`` (default), draw a legend.
    median : bool, optional
        If ``True``, use the median instead of the mean within each bin.
        Default is ``False``.
    restore_mean : bool, optional
        If ``True``, restore the overall mean after absorbing fixed effects.
        Default is ``False``.
    title : str, optional
        Figure title.
    include45 : bool, optional
        If ``True``, overlay a 45-degree line and force equal axis ranges.
        Default is ``False``.
    include0 : bool, optional
        If ``True``, overlay a horizontal line at zero and expand the y-axis
        if necessary to include it.  Default is ``False``.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    pandas.DataFrame
        DataFrame of binned x and y values used in the scatter plot.
    """

    if control is None:
        control = []
    if absorb is None:
        absorb = []

    if labels is None:
        labels = {}
    if bin_kwargs is None:
        bin_kwargs = {}
    if raw_kwargs is None:
        raw_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    matplotlib.rcParams.update({"font.size": label_font})

    if isinstance(yvars, str):
        yvars = [yvars]
    combined = len(yvars) > 1

    if isinstance(absorb, str):
        absorb = [absorb]

    absorb_list = []
    for item in absorb:
        if isinstance(item, str):
            absorb_list.append(item)
        else:
            absorb_list += item

    keep_list = [xvar] + yvars + control + absorb_list
    if wvar is not None:
        keep_list.append(wvar)
    if fit_var is not None:
        keep_list.append(fit_var)

    df = df_in.reset_index()
    df = df[keep_list]
    df = df.dropna()

    bin_kwargs_new = {
        "alpha": 1.0,
        "edgecolor": "black",
        "zorder": 2,
        "label": "Binscatter",
    }

    line_kwargs_new = {
        "linewidth": 2,
        "zorder": 1,
        "color": "firebrick",
        "label": "_nolabel",
    }

    raw_kwargs_new = {
        "marker": "o",
        "color": "cornflowerblue",
        "alpha": 0.5,
        "edgecolor": "black",
        "label": "Raw Data",
    }

    if plot_raw_data:
        if bin_scale is None:
            bin_scale = 100.0

        bin_kwargs_new.update(
            {
                "marker": "*",
                "color": "firebrick",
            }
        )
    else:
        if bin_scale is None:
            bin_scale = 50.0

        bin_kwargs_new.update(
            {
                "marker": "o",
                "color": "cornflowerblue",
            }
        )

        raw_kwargs_new = {}

    if combined:
        line_kwargs_new.update(
            {
                "linestyle": "--",
            }
        )
    else:
        raw_kwargs_new.update(
            {
                "label": "Raw Data",
            }
        )

    bin_kwargs_new.update(bin_kwargs)
    raw_kwargs_new.update(raw_kwargs)
    line_kwargs_new.update(line_kwargs)

    if wvar is None:
        weights = np.ones(len(df))
    else:
        weights = df[wvar].astype(np.float64).values

    if control or absorb:
        for this_var in [xvar] + yvars:
            this_mean = stats.weighted_mean(df[this_var].values, weights)

            if absorb:
                df[this_var] = dt.absorb(
                    df, absorb, this_var, weight_var=wvar, restore_mean=restore_mean
                )

            if control:
                fr = dt.regression(df, this_var, control, weight_var=wvar)
                df[this_var] = np.nan
                df.loc[fr.ix, this_var] = fr.results.resid

            df[this_var] += this_mean

    fig, ax = plt.subplots()
    for iy, yvar in enumerate(yvars):
        if combined:
            assert not plot_raw_data
            this_color = "C{:d}".format(iy)
            bin_kwargs_new.update(
                {
                    "color": this_color,
                    "label": labels.get(yvar, yvar),
                }
            )
            line_kwargs_new.update(
                {
                    "color": this_color,
                }
            )

        by_bin = dt.compute_binscatter(
            df, yvar, xvar, wvar=wvar, n_bins=n_bins, bins=bins, median=median
        )

        if plot_line and (fit_var is None):
            fr = dt.regression(df, yvar, [xvar], weight_var=wvar)
            df.loc[fr.ix, yvar + "_fit"] = fr.results.fittedvalues

        if plot_line:
            x = df[xvar].values
            fit_col = fit_var if fit_var is not None else yvar + "_fit"
            y_fit = df[fit_col].values
            imin = np.argmin(x)
            imax = np.argmax(x)
            x_line = np.array((x[imin], x[imax]))
            y_line = np.array((y_fit[imin], y_fit[imax]))
            ax.plot(x_line, y_line, **line_kwargs_new)

        if plot_raw_data:
            weights *= raw_scale / np.mean(weights)

            ax.scatter(df[xvar].values, df[yvar].values, s=weights, **raw_kwargs_new)

        ax.scatter(
            by_bin[xvar].values,
            by_bin[yvar].values,
            s=bin_scale,
            **bin_kwargs_new,
        )

    plt.xlabel(labels.get(xvar, xvar))

    if use_legend:
        plt.legend(fontsize=legend_font)

    if not combined:
        yvar = yvars[0]
        plt.ylabel(labels.get(yvar, yvar))

    if xlim is None:
        if plot_raw_data:
            xlim = stats.weighted_quantile(df[xvar].values, weights, [0.005, 0.995])
        else:
            bin_min = np.amin(by_bin[xvar].values)
            bin_max = np.amax(by_bin[xvar].values)
            tot_range = bin_max - bin_min
            xlim = (bin_min - 0.05 * tot_range, bin_max + 0.05 * tot_range)

    if ylim is None:
        if plot_raw_data:
            ylim = stats.weighted_quantile(df[yvar].values, weights, [0.005, 0.995])
        else:
            bin_min = np.amin(by_bin[yvar].values)
            bin_max = np.amax(by_bin[yvar].values)
            tot_range = bin_max - bin_min
            ylim = (bin_min - 0.1 * tot_range, bin_max + 0.1 * tot_range)

    if include0 or include45:
        assert (xlim != "default") and (ylim != "default")

    if include45:
        assert not include0
        plot_lb = min(xlim[0], ylim[0])
        plot_ub = max(xlim[1], ylim[1])
        xlim = (plot_lb, plot_ub)
        ylim = (plot_lb, plot_ub)
        plt.plot([plot_lb, plot_ub], [plot_lb, plot_ub], "k--")
    elif include0:
        plt.plot(xlim, np.zeros(2), "k--")
        if ylim[0] > 0.0:
            ylim = (-0.1 * ylim[1], ylim[1])
        elif ylim[1] < 0.0:
            ylim = (ylim[0], -0.1 * ylim[0])

    if (
        (xlim is not None)
        and (xlim != "default")
        and all([np.isfinite(val) for val in xlim])
    ):
        plt.xlim(xlim)

    if (
        (ylim is not None)
        and (ylim != "default")
        and all([np.isfinite(val) for val in ylim])
    ):
        plt.ylim(ylim)

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)

    plt.close(fig)

    return by_bin


def scatter(
    df,
    yvar,
    xvar,
    labels=None,
    multicolor=False,
    cmap_name="plasma",
    color="C0",
    include45=False,
    filepath=None,
):
    """Create a simple scatter plot of two columns from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the variables to plot.
    yvar : str
        Column name for the y-axis variable.
    xvar : str
        Column name for the x-axis variable.
    labels : dict, optional
        Mapping from column name to axis label string.
    multicolor : bool, optional
        If ``True``, color each point according to its position in the
        DataFrame using *cmap_name*.  Default is ``False``.
    cmap_name : str, optional
        Matplotlib colormap name used when *multicolor* is ``True``.
        Default is ``'plasma'``.
    color : str, optional
        Uniform color for all points when *multicolor* is ``False``.
        Default is ``'C0'``.
    include45 : bool, optional
        If ``True``, overlay a 45-degree dashed line.  Default is ``False``.
    filepath : str, optional
        File path to save the figure.  Displays interactively when ``None``.
    """

    if labels is None:
        labels = {}

    if multicolor:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.0, 1.0, len(df)))[::-1]
    else:
        colors = color

    fig = plt.figure()
    plt.scatter(df[xvar], df[yvar], c=colors, edgecolor="gray", alpha=0.75)

    if include45:
        plot_lb, plot_ub = get_45_bounds(df, xvar, yvar)
        plt.plot([plot_lb, plot_ub], [plot_lb, plot_ub], "k--")

    plt.xlabel(labels.get(xvar, xvar))
    plt.ylabel(labels.get(yvar, yvar))
    plt.tight_layout()

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
    plt.close(fig)


def state_scatter_inner(ax, this_df, yvar, xvar):
    """Annotate each data point with the corresponding state name.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the annotations.
    this_df : pandas.DataFrame
        DataFrame indexed by state name with columns for *xvar* and *yvar*.
    yvar : str
        Column name for the y-coordinate of each annotation.
    xvar : str
        Column name for the x-coordinate of each annotation.

    Returns
    -------
    None
    """
    for ii, state in enumerate(this_df.index):
        ax.annotate(state, (this_df.loc[state, xvar], this_df.loc[state, yvar]))
    return None


def get_colors(n, cmap_name, startval=0.0, endval=1.0):
    """Sample *n* evenly spaced colors from a matplotlib colormap.

    Parameters
    ----------
    n : int
        Number of colors to return.
    cmap_name : str
        Name of the matplotlib colormap (e.g. ``'viridis'``, ``'plasma'``).
    startval : float, optional
        Starting position in the colormap in the range [0, 1].  Default is
        0.0.
    endval : float, optional
        Ending position in the colormap in the range [0, 1].  Default is
        1.0.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Array of RGBA color values.
    """
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(startval, endval, n))
    return colors
