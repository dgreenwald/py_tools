import matplotlib
import os

if os.environ.get('USE_MATPLOTLIB_AGG', 0):
    matplotlib.use('Agg')

# import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.style as plt_style
import numpy as np
import pandas as pd
from scipy.stats import norm

# from py_tools.data import clean
from py_tools import data as dt, stats

pd.plotting.register_matplotlib_converters()

def set_fontsizes(ax, fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def save_hist(vals, path, **kwargs):

    h = np.histogram(vals, **kwargs)
    dt.to_pickle(h, path)

def save_hist_npy(vals, base_path, **kwargs):

    hist, bin_edges = np.histogram(vals, **kwargs)
    np.save(base_path + 'hist.npy', hist) 
    np.save(base_path + 'bin_edges.npy', bin_edges) 

def load_hist_npy(base_path):

    hist = np.load(base_path + 'hist.npy') 
    bin_edges = np.load(base_path + 'bin_edges.npy') 

    return hist, bin_edges

def two_axis(df_in, var1, var2, filepath=None, loc1='upper left', 
             loc2='upper right', loc_single=None, legend_font=10, label_font=12,
             normalize=False, color1=None, color2=None, colors=None, flip1=False,
             flip2=False, legend=True,
             single_legend=False, print_legend_axis=True, labels=None,
             leglabels=None, drop=True, kwargs1=None, kwargs2=None, format_dates=False,
             title=None, figsize=None, style=None,
             savefig_kwargs=None):

    if colors is None: colors = {}
    if labels is None: labels = {}
    if leglabels is None: leglabels = {}
    if kwargs1 is None: kwargs1 = {}
    if kwargs2 is None: kwargs2 = {}
    if savefig_kwargs is None: savefig_kwargs = {}
    
    if style is not None:
        plt_style.use(style)
    
    kwargs1_copy = kwargs1.copy()
    kwargs1 = {'linewidth' : 2,}
    kwargs1.update(kwargs1_copy)
    
    kwargs2_copy = kwargs2.copy()
    kwargs2 = {'linewidth' : 2, 'marker' : 'o', 'markevery' : 4,
               'fillstyle' : 'none', 'markersize' : 5, 'mew' : 1.5}
    kwargs2.update(kwargs2_copy)

    matplotlib.rcParams.update({'font.size' : label_font})
    
    df = df_in[[var1, var2]].copy()
    if drop:
        df = df.dropna()

    for these_kwargs in [kwargs1, kwargs2]:
        if these_kwargs.get('marker', None) is not None:
            if these_kwargs.get('markevery', None) is None:
                these_kwargs['markevery'] = np.round(len(df) / 20)

    if color1 is None:
        color1 = colors.get(var1, '#1f77b4')
    if color2 is None:
        color2 = colors.get(var2, '#ff7f0e')

    fig, ax1 = plt.subplots(figsize=figsize)

    label1 = labels.get(var1, var1)
    label2 = labels.get(var2, var2)

    leglabel1 = leglabels.get(var1, label1)
    leglabel2 = leglabels.get(var2, label2)
    if print_legend_axis:
        leglabel1 = leglabel1 + ' (left)'
        leglabel2 = leglabel2 + ' (right)'

    if flip1:
        line1 = ax1.plot(df.index, -df[var1], label=('(-1) x ' + leglabel1), color=color1, **kwargs1)        
        # (-df[var1]).plot(ax=ax1, linewidth=2, label=('(-1) x ' + leglabel1), color=color1)
    else:
        line1 = ax1.plot(df.index, df[var1], label=leglabel1, color=color1, **kwargs1)
        # df[var1].plot(ax=ax1, linewidth=2, label=leglabel1, color=color1)

    ax2 = ax1.twinx()
    if flip2:
        line2 = ax2.plot(df.index, -df[var2], label=('(-1) x ' + leglabel2), 
                         color=color2, **kwargs2)

        # (-df[var2]).plot(ax=ax2, linestyle='-', linewidth=2, label=('(-1) x ' + leglabel2), 
                         # color=color2, marker=mark2, fillstyle='none', markersize=5, 
                         # mew=1.5, markevery=markevery)
    else:
        line2 = ax2.plot(df.index, df[var2], label=leglabel2, color=color2, **kwargs2)
        # df[var2].plot(ax=ax2, linestyle='-', linewidth=2, label=leglabel2, color=color2,
                      # marker='o', fillstyle='none', markersize=5, mew=1.5, markevery=markevery)

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

        ylim_norm = np.array(np.minimum(ax1_ylim_norm[0], ax2_ylim_norm[0]),
                             np.maximum(ax1_ylim_norm[1], ax2_ylim_norm[1]))

        ax1_ylim_new = df[var1].std() * ylim_norm + df[var1].mean()
        ax2_ylim_new = df[var2].std() * ylim_norm + df[var2].mean()

        # ax1_ylim_new = tuple([
            # df[var1].std() * (val - df[var1].mean()) / df[var1].std() 
            # + df[var2].mean()
            # for val in ax1_ylim
        # ])

        # ax2_ylim = tuple([
            # df[var2].std() * (val - df[var1].mean()) / df[var1].std() 
            # + df[var2].mean()
            # for val in ax1_ylim
        # ])

        ax1.set_ylim(ax1_ylim_new)
        ax2.set_ylim(ax2_ylim_new)

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
    
    if invert_list is None:
        invert_list = len(var_list) * [False]
    
    fig = plt.figure()
    
    # for this_var, invert in zip(var_list, invert_list):
    for this_var in var_list:
        
        x = df[this_var].values.copy()
        x -= np.mean(x)
        x /= np.std(x)
        
        if this_var in invert_list:
            x *= -1
            invert_str = '(-1) x '
        else:
            invert_str = ''
            
        plt.plot(df.index, x, label=invert_str + this_var)

    plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)
    
    return None

def hist(df_in, var, label=None, xlabel=None, ylabel=None, wvar=None, 
         bins=None, xlim=None, ylim=None, filepath=None,
         legend_font=10, label_font=12, copy_path=None, x_vertline=None,
         vertline_kwargs=None, **kwargs):

    if vertline_kwargs is None: vertline_kwargs = {}

    df = dt.clean(df_in, [var, wvar])

    if var not in df or len(df) == 0:
        return False

    if wvar is not None:
        w = df[wvar].values
    else:
        w = np.ones(len(df))

    # Normalize
    if matplotlib.__version__ == '2.0.2':
        kwargs['normed'] = True
    else:
        kwargs['density'] = True

    # TODO: could use kwargs for some of these
    fig = plt.figure()
    matplotlib.rcParams.update({'font.size' : label_font})
    plt.hist(df[var].values, bins=bins, alpha=0.5, edgecolor='black',
             weights=w, label=label, **kwargs)
    
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

    this_list = [var]
    if wvar is not None:
        this_list.append(wvar)
        
    _df = df[this_list].copy()
    if wvar is None:
        wvar = '_ONES_'
        _df[wvar] = 1.0
    
    _df['bin'] = pd.cut(_df[var], bins, labels=bins[:-1])
    hist = _df.groupby('bin')[wvar].sum()
    hist /= np.sum(hist)

    return hist

def multi_hist(dfs, labels=None, xvar=None, xvars=None, bins=None, wvar=None, wvars=None,
                filepath=None, xlabel=None, ylabel=None, xlim=None,
                ylim=None, legend_font=10, label_font=12, copy_paths=None,
                colors=None, edgecolor='black', alpha=0.5, use_bar=False, 
                kwarg_list=None, x_vertline=None,
                vertline_kwargs=None, topcode=False, bottomcode=False, 
                density=True,
                **kwargs):
    """Plots double histogram overlaying var1 from df1 and var2 from df2

    Arguments:
    """

    if vertline_kwargs is None: vertline_kwargs = {}

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
        wvar = '_count'
        
    # Bound at edges of bins
    if (xlim is None) and (bins is not None):
        xlim = bins[[0, -1]]
        
    # Normalize
    if density and (not use_bar):
        if matplotlib.__version__ == '2.0.2':
            kwargs['normed'] = True
        else:
            kwargs['density'] = True
        
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
        
        if wvars[ii] == '_count':
            df['_count'] = 1.0
        else:
            df[wv] = dfs[ii][wv]
            
        df = dt.clean(df, [xv, wv])

        if topcode:
            df[xv] = np.minimum(df[xv], bins[-1])
            
        if bottomcode:
            df[xv] = np.maximum(df[xv], bins[0])
    
        _dfs.append(df)
        
    fig = plt.figure()
    matplotlib.rcParams.update({'font.size' : label_font})
        
    if use_bar:
        
        assert bins is not None
        
        # TOD0: 
        bins = np.array(bins)
        bin_width = bins[1:] - bins[:-1]

        for ii in ilist:
            
            hist = compute_hist(_dfs[ii], xvars[ii], bins, wvar=wvars[ii])
            
            plt.bar(np.array(hist.index), hist.values, width=bin_width,
                    alpha=alpha, edgecolor=edgecolor, align='edge',
                    label=str(labels[ii]), color=colors[ii], **kwarg_list[ii]
                    )

    else:
        
        for ii in ilist:

            plt.hist(_dfs[ii][xvars[ii]].values, bins=bins, alpha=alpha, edgecolor=edgecolor,
                     weights=_dfs[ii][wvars[ii]].copy(), label=str(labels[ii]), color=colors[ii], **kwarg_list[ii])

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
        
    plt.legend(fontsize=legend_font)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    if copy_paths is not None:
        for ii in ilist:
            save_hist(_dfs[ii][xvars[ii]].values, copy_paths[ii], density=True, bins=bins, weights=_dfs[ii][wvars[ii]].values)

    return True

def double_hist(df1, df2=None, label1=None, label2=None, var=None,
                var1=None, var2=None, bins=None, wvar=None, wvar1=None,
                wvar2=None, filepath=None, xlabel=None, ylabel=None, xlim=None,
                ylim=None, legend_font=10, label_font=12, copy_path1=None,
                copy_path2=None, color1=None, color2=None, edgecolor='black', 
                alpha=0.5, use_bar=False, labels=None, kwargs1=None, kwargs2=None, x_vertline=None,
                vertline_kwargs=None, topcode=False, bottomcode=False, **kwargs):
    """Plots double histogram overlaying var1 from df1 and var2 from df2

    Arguments:
    """

    if labels is None: labels = {}
    if kwargs1 is None: kwargs1 = {}
    if kwargs2 is None: kwargs2 = {}
    if vertline_kwargs is None: vertline_kwargs = {}

    kwargs1.update(kwargs)
    kwargs2.update(kwargs)

    if df2 is None:
        df2 = df1

    if var is not None:
        assert var1 is None and var2 is None
        var1 = var
        var2 = var
        
    if wvar is None:
        wvar = '_count'
        
    if wvar1 is None:
        wvar1 = wvar
        
    if wvar2 is None:
        wvar2 = wvar

    if label1 is None:
        label1 = labels.get(var1, var1)

    if label2 is None:
        label2 = labels.get(var2, var2)
        
    # if edgecolor is None:
        # edgecolor = 'black'
    # if edgecolor1 is None:
        # edgecolor1 = edgecolor
    # if edgecolor2 is None:
        # edgecolor2 = edgecolor

    # Normalize
    if matplotlib.__version__ == '2.0.2':
        kwargs['normed'] = True
    else:
        kwargs['density'] = True
        
    _df1 = df1[[var1]].copy()
    _df2 = df2[[var2]].copy()
    
    if wvar1 == '_count':
        _df1['_weight1'] = 1.0
    else:
        _df1['_weight1'] = df1[wvar1]
        
    if wvar2 == '_count':
        _df2['_weight2'] = 1.0
    else:
        _df2['_weight2'] = df2[wvar2]

    # _df1 = dt.clean(df1, [var1, wvar1])
    # _df2 = dt.clean(df2, [var2, wvar2])
    
    wvar1 = '_weight1'
    wvar2 = '_weight2'
    
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
    
    # if wvar1 == '_count':
    #     _df1[wvar1] = 1.0
    # if wvar2 == '_count':
    #     _df2[wvar2] = 1.0

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
    matplotlib.rcParams.update({'font.size' : label_font})
    
    kwargs1.update(kwargs)
    kwargs2.update(kwargs)

    if use_bar:
        
        assert bins is not None
        
        bin_width = bins[1] - bins[0]

        hist1 = compute_hist(_df1, var1, bins, wvar=wvar1)
        hist2 = compute_hist(_df2, var2, bins, wvar=wvar2)

        plt.bar(np.array(hist1.index), hist1.values, width=bin_width,
                alpha=alpha, edgecolor=edgecolor, align='edge',
                label=str(label1), color=color1, **kwargs1)
        plt.bar(np.array(hist2.index), hist2.values, width=bin_width,
                alpha=alpha, edgecolor=edgecolor, align='edge',
                label=str(label2), color=color2, **kwargs2)

    else:

        plt.hist(_df1[var1].values, bins=bins, alpha=alpha, edgecolor=edgecolor,
                 weights=w1, label=str(label1), color=color1, **kwargs1)
        plt.hist(_df2[var2].values, bins=bins, alpha=alpha, edgecolor=edgecolor,
                 weights=w2, label=str(label2), color=color2, **kwargs2)

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
        
    plt.legend(fontsize=legend_font)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    if copy_path1 is not None:
        save_hist(_df1[var1].values, copy_path1, density=True, bins=bins, weights=w1)
    if copy_path2 is not None:
        save_hist(_df2[var2].values, copy_path1, density=True, bins=bins, weights=w2)

    return True

def var_irfs(irfs, var_list, shock_list=None, titles=None, filepath=None,
             n_per_row=None, plot_scale=3):

    if titles is None: titles = {}

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
            plt.subplot(n_rows, n_per_row, Ny*ishock + iy + 1)

            plt.plot(np.zeros(Nirf), color='gray', linestyle=':')
            plt.plot(center[:, iy, ishock], color='blue')
            plt.plot(bands[0, :, iy, ishock], color='black', linestyle='--')
            plt.plot(bands[1, :, iy, ishock], color='black', linestyle='--')

            plt.xlim((0, Nirf - 1))

            var_title = titles.get(var_list[iy], var_list[iy])
            shock_title = titles.get(shock_list[ishock], shock_list[ishock])
            plt.title('{0} to {1}'.format(var_title, shock_title))

# plt.show()
    if filepath is None:
        plt.show()
    else:
        fig.set_size_inches((plot_scale * n_per_row, plot_scale * n_rows))
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)

    return None 

def plot_series(df_in, var_names, filepath=None, directory=None, filename=None, labels=None,
                linestyles=None, markers=None, colors=None, markevery=8,
                markersize=5, mew=2, fillstyle='none', fontsize=12,
                # plot_type='pdf', 
                plot_type=None,
                ylabel=None, sample='outer', title=None, 
                single_legend=True, vertline_ix=None,
                vertline_kwargs=None, linewidths=None, ylim=None, dpi=None):

    if labels is None: labels = {}
    if linestyles is None: linestyles = {}
    if markers is None: markers = {}
    if colors is None: colors = {}
    if vertline_kwargs is None: vertline_kwargs = {}
    if linewidths is None: linewidths = {}

    matplotlib.rcParams.update({'font.size' : fontsize})
    
    if (directory is not None) or (filename is not None) or (plot_type is not None):
        print("Switch to new filepath input format")
        raise Exception

    # if directory != '' and directory[-1] != '/':
    #     directory += '/'

    # if filename is None:
    #     filename = '_'.join(var_names)

    fig = plt.figure()

    if sample == 'outer':
        ix = np.any(pd.notnull(df_in[var_names]), axis=1)
    elif sample == 'inner':
        ix = np.all(pd.notnull(df_in[var_names]), axis=1)
    else:
        raise Exception

    df = df_in.loc[ix, var_names].copy()

    for var in var_names:

        label = labels.get(var, var)
        linestyle = linestyles.get(var, '-')
        color = colors.get(var, None)
        linewidth=linewidths.get(var, 2)

        marker = markers.get(var, None)

#        if label is None:
#            plt.plot(
#                df.index, df[var],
#                linewidth=linewidth, linestyle=linestyle, marker=marker,
#                markevery=markevery, markersize=markersize, mew=mew, color=color
#            )
#        else:
        plt.plot(
            df.index, df[var].values,
            linewidth=linewidth, linestyle=linestyle, label=label, marker=marker,
            markevery=markevery, markersize=markersize, mew=mew, color=color
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

def projection(x, se, var_titles, shock_title, p=0.9, n_per_row=4, plot_size=3.0,
               out_dir=None, label=None, shock_name=None): 
    """Plot impulse responses from a local projection estimation.

    x: Ny x Nt vector of coefficients (one per LHS variable)
    se: Ny x Nt vector of standard errors (one per LHS variable)
    var_titles: Ny x 1 list of names of LHS variables (one per LHS variable)
    shock_title: name of shock
    p: probability span of confidence bands (i.e., 0.9 = 90%)
    n_per_row: number of plots per row
    plot_size: inches per side of each plot
    out_dir: directory to save plots in (will display to screen if not provided)
    label: additional title to put in plot name
    shock_name: optional shorter name for file
    """

    Ny, Nt = x.shape
    n_rows = (Ny - 1) // n_per_row + 1 # number of rows needed
    
    if n_rows == 1:
        n_per_row = Ny

    # Get z-score for p-value
    z_star = -norm.ppf(0.5 * (1.0 - p))

    # Plot
    fig = plt.figure()
    for iy in range(Ny):

        plt.subplot(n_rows, n_per_row, iy + 1)

        plt.plot(np.arange(Nt), np.zeros(x[iy, :].shape), linestyle=':', color='gray') 
        plt.plot(np.arange(Nt), x[iy, :], linestyle='-', color='blue') 
        plt.plot(np.arange(Nt), x[iy, :] + z_star * se[iy, :], linestyle='--', color='black') 
        plt.plot(np.arange(Nt), x[iy, :] - z_star * se[iy, :], linestyle='--', color='black')

        plt.title('{0} to {1}'.format(var_titles[iy], shock_title))

        plt.xlim((0, Nt - 1))
        plt.xlabel('Quarters')

    fig.set_size_inches((plot_size * n_per_row, plot_size * n_rows))
    plt.tight_layout()

    if out_dir is None:

        plt.show()

    else:

        if label is None:
            prefix = ''
        else:
            prefix = label + '_'

        if shock_name is None:
            shock_name = shock_title

        plt.savefig('{0}/{1}{2}_projections.pdf'.format(out_dir, prefix, shock_name))

    plt.close(fig)

    return None

def get_45_bounds(df, xvar, yvar, margin=0.05):
    
    min_val = np.amin(df[[yvar, xvar]].values)
    max_val = np.amax(df[[yvar, xvar]].values)
    dist = max_val - min_val
    
    plot_lb = min_val - margin * dist
    plot_ub = max_val + margin * dist
    
    return plot_lb, plot_ub

def binscatter(df_in, yvars, xvar, wvar=None, fit_var=None, labels=None, n_bins=20, bins=None,
               filepath=None, xlim=None, ylim=None, plot_line=True, 
               control=None, absorb=None, bin_scale=None, raw_scale=10.0,
               plot_raw_data=False, bin_kwargs=None, raw_kwargs=None, line_kwargs=None,
               legend_font=10, label_font=12, use_legend=True, median=False,
               restore_mean=False, title=None, include45=False, include0=False,
               **kwargs):

    if control is None: control = []
    if absorb is None: absorb = []

    if labels is None: labels = {}
    if bin_kwargs is None: bin_kwargs = {}
    if raw_kwargs is None: raw_kwargs = {}
    if line_kwargs is None: line_kwargs = {}
        
    matplotlib.rcParams.update({'font.size' : label_font})
    
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
        'alpha' : 1.0,
        'edgecolor' : 'black',
        'zorder' : 2,
        'label' : 'Binscatter',
    }
    
    line_kwargs_new = {
            'linewidth' : 2,
            'zorder' : 1,
            'color' : 'firebrick',
            'label' : '_nolabel',
            }
    
    raw_kwargs_new = {
        'marker' : 'o',
        'color' : 'cornflowerblue',
        'alpha' : 0.5,
        'edgecolor' : 'black',
        'label' : 'Raw Data',
    }

    if plot_raw_data:

        if bin_scale is None:
            bin_scale = 100.0

        bin_kwargs_new.update({
            'marker' : '*',
            'color' : 'firebrick',
        })
    else:

        if bin_scale is None:
            bin_scale = 50.0

        bin_kwargs_new.update({
            'marker' : 'o',
            'color' : 'cornflowerblue',
        })

        raw_kwargs_new = {}
        
    if combined:
        line_kwargs_new.update({
                'linestyle' : '--',
                })
    else:
        raw_kwargs_new.update({
                'label' : 'Raw Data',
                })

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
                df[this_var] = dt.absorb(df, absorb, this_var, weight_var=wvar, 
                                    restore_mean=restore_mean)

            if control:
                fr = dt.regression(df, this_var, control, weight_var=wvar)
                df[this_var] = np.nan
                df.loc[fr.ix, this_var] = fr.results.resid

            df[this_var] += this_mean

    fig, ax = plt.subplots()
    for iy, yvar in enumerate(yvars):
        
        if combined:
            assert not plot_raw_data
            this_color = 'C{:d}'.format(iy)
            bin_kwargs_new.update({
                    'color' : this_color,
                    'label' : labels.get(yvar, yvar),
                    })
            line_kwargs_new.update({
                    'color' : this_color,
                    })
        
        by_bin = dt.compute_binscatter(df, yvar, xvar, wvar=wvar, n_bins=n_bins, 
                                       bins=bins, median=median)
        
        if plot_line and (fit_var is None):
            fr = dt.regression(df, yvar, [xvar], weight_var=wvar)
            df.loc[fr.ix, yvar + '_fit'] = fr.results.fittedvalues
        
        if plot_line:
            x = df[xvar].values
            y_fit = df[yvar + '_fit'].values
            imin = np.argmin(x)
            imax = np.argmax(x)
            x_line = np.array((x[imin], x[imax]))
            y_line = np.array((y_fit[imin], y_fit[imax]))
            ax.plot(x_line, y_line, **line_kwargs_new)
#            ax.plot(df[xvar], df[yvar + '_fit'], color=line_color, linewidth=2,
#                    linestyle=linestyle, zorder=0)
                        
        if plot_raw_data:
                
            weights *= (raw_scale / np.mean(weights))
            
            ax.scatter(df[xvar].values, df[yvar].values, 
                       s=weights, **raw_kwargs_new
                    )
            
        ax.scatter(by_bin[xvar].values, by_bin[yvar].values, 
                   s=bin_scale, **bin_kwargs_new,
                    )
        
    # if include45:
    #     assert not include0
    #     plot_lb, plot_ub = get_45_bounds(by_bin, xvar, yvar)
    #     plt.plot([plot_lb, plot_ub], [plot_lb, plot_ub], 'k--')
    # elif include0:
    #     plot_lb = np.amin(by_bin[xvar].values)
    #     plot_ub = np.amin(by_)
    
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
        assert (xlim != 'default') and (ylim != 'default')
        
    if include45:
        assert not include0
        plot_lb = min(xlim[0], ylim[0])
        plot_ub = max(xlim[1], ylim[1])
        xlim = (plot_lb, plot_ub)
        ylim = (plot_lb, plot_ub)
        plt.plot([plot_lb, plot_ub], [plot_lb, plot_ub], 'k--')
    elif include0:
        plt.plot(xlim, np.zeros(2), 'k--')
        if ylim[0] > 0.0:
            ylim = (-0.1 * ylim[1], ylim[1])
        elif ylim[1] < 0.0:
            ylim = (ylim[0], -0.1 * ylim[0])
        
    if (xlim is not None) and (xlim != 'default') and all([np.isfinite(val) for val in xlim]):
        plt.xlim(xlim)   
        
    if (ylim is not None) and (ylim != 'default') and all([np.isfinite(val) for val in ylim]):
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

def scatter(df, yvar, xvar, labels=None,
            multicolor=False, cmap_name='plasma', color='C0', 
            include45=False, filepath=None):

    if labels is None: labels = {}
    
    if multicolor:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.0, 1.0, len(df)))[::-1]
    else:
        colors = color
    
    fig = plt.figure()
    plt.scatter(df[xvar], df[yvar], c=colors, edgecolor='gray', alpha=0.75)
    
    if include45:
        plot_lb, plot_ub = get_45_bounds(df, xvar, yvar)
        plt.plot([plot_lb, plot_ub], [plot_lb, plot_ub], 'k--')
        # plt.xlim((plot_lb, plot_ub))
    
    plt.xlabel(labels[xvar])
    plt.ylabel(labels[yvar])
    plt.tight_layout()
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
    plt.close(fig)
    
def state_scatter_inner(ax, this_df, yvar, xvar):
    for ii, state in enumerate(this_df.index):
        ax.annotate(state, (this_df.loc[state, xvar], this_df.loc[state, yvar]))
    return None

def get_colors(n, cmap_name, startval=0.0, endval=1.0):
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(startval, endval, n))
    return colors

# From TomAugsburger

#def add_rec_bars(ax, dates=None, alpha=0.25, color='k'):
#
#    if dates is None:
#        dates = misc.load('nber_dates')
#        # dates = pd.read_csv('/Users/tom/bin/rec_dates.csv',
#                            # parse_dates=['Peak', 'Trough'])
#
#    xbar = ax.get_xlim()
#    y1, y2 = ax.get_ylim()
#    
#    # First convert to pandas Period
#    foo = matplotlib.dates.num2date(ax.get_xlim()[0])
#    
#    for row in dates.iterrows():
#        x = row[1]
##        y1, y2 = ax.get_ylim()
#        if x[0] > xbar[0] or x[1] < xbar[1]:
#            ax.fill_between([x[0], x[1]], y1=y1, y2=y2, alpha=alpha, color=color)
#
#    return ax
