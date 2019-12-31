import matplotlib
import os

if os.environ.get('USE_MATPLOTLIB_AGG', 0):
    matplotlib.use('Agg')

# import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# from py_tools.data import clean
from py_tools import data as dt, stats

pd.plotting.register_matplotlib_converters()

def save_hist(vals, path, **kwargs):

    h = np.histogram(vals, **kwargs)
    dt.to_pickle(h, path)

def two_axis(df_in, var1, var2, filepath=None, loc1='upper left', 
             loc2='upper right', loc_single=None, legend_font=10, label_font=12,
             normalize=False, color1='#1f77b4', color2='#ff7f0e', flip1=False,
             flip2=False, legend=True,
             single_legend=False, print_legend_axis=True, labels={},
             leglabels={}, drop=True, kwargs1={}, kwargs2={}):
    
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
                these_kwargs['markevery'] = math.round(len(df) / 20)

    fig, ax1 = plt.subplots()

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

    if filepath is not None:
        plt.tight_layout()
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    return None

def normalized(df, var_list, filepath=None, invert_list=[]):
    
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
         vertline_kwargs={}, **kwargs):

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
        save_hist(df[var].values, copy_path, density=True, bins=bins, weights=w)

    return True

def compute_hist(df, var, bins, wvar=None):

    df['bin'] = pd.cut(df[var], bins, labels=bins[:-1])
    hist = df.groupby('bin')[wvar].sum()
    hist /= np.sum(hist)

    return hist

def double_hist(df1, df2=None, label1=None, label2=None, var=None,
                var1=None, var2=None, bins=None, wvar=None, wvar1=None,
                wvar2=None, filepath=None, xlabel=None, ylabel=None, xlim=None,
                ylim=None, legend_font=10, label_font=12, copy_path1=None,
                copy_path2=None, color1=None, color2=None, edgecolor='black', 
                alpha=0.5, use_bar=False, labels={}, kwargs1={}, kwargs2={}, x_vertline=None,
                vertline_kwargs={}, **kwargs):
    """Plots double histogram overlaying var1 from df1 and var2 from df2

    Arguments:
    """

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

    _df1 = dt.clean(df1, [var1, wvar1])
    _df2 = dt.clean(df2, [var2, wvar2])
    
    if wvar1 == '_count':
        _df1[wvar1] = 1.0
    if wvar2 == '_count':
        _df2[wvar2] = 1.0

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

def var_irfs(irfs, var_list, shock_list=None, titles={}, filepath=None,
             n_per_row=None, plot_scale=3):

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

def plot_series(df_in, var_names, directory='', filename=None, labels={},
                linestyles={}, markers={}, colors={}, markevery=8,
                markersize=5, mew=2, fillstyle='none', fontsize=12,
                plot_type='pdf', ylabel=None, sample='outer', title=None,
                save=True, single_legend=True, vertline_ix=None,
                vertline_kwargs={}, linewidths={}, ylim=None):

    matplotlib.rcParams.update({'font.size' : fontsize})

    if directory != '' and directory[-1] != '/':
        directory += '/'

    if filename is None:
        filename = '_'.join(var_names)

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
    if save:
        plt.savefig('{0}{1}.{2}'.format(directory, filename, plot_type))
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

def binscatter(df_in, yvars, xvar, wvar=None, fit_var=None, labels={}, n_bins=20, 
               filepath=None, xlim=None, ylim=None, plot_line=True, 
               control=[], absorb=[], bin_scale=None, raw_scale=10.0,
               plot_raw_data=False, bin_kwargs={}, raw_kwargs={}, line_kwargs={},
               legend_font=10, label_font=12, use_legend=True,
               **kwargs):
        
    matplotlib.rcParams.update({'font.size' : label_font})
    
    if isinstance(yvars, str):
        yvars = [yvars]
    combined = len(yvars) > 1
        
    if isinstance(absorb, str):
        absorb = [absorb]
    
    keep_list = [xvar] + yvars + control + absorb
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
                                    restore_mean=False)

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
        
        by_bin = dt.compute_binscatter(df, yvar, xvar, wvar=wvar, n_bins=n_bins)
        
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
    
    plt.xlabel(labels.get(xvar, xvar))
    
    if use_legend:
        plt.legend(fontsize=legend_font)
        
    if not combined:
        yvar = yvars[0]
        plt.ylabel(labels.get(yvar, yvar))
    
    if xlim != 'default':
        if xlim is not None:
            plt.xlim(xlim)
        else:
            if plot_raw_data:
                xlim = stats.weighted_quantile(df[xvar].values, df[wvar].values, [0.005, 0.995])
                plt.xlim(xlim)
            else:
                bin_min = np.amin(by_bin[xvar].values)
                bin_max = np.amax(by_bin[xvar].values)
                tot_range = bin_max - bin_min
                xlim = (bin_min - 0.05 * tot_range, bin_max + 0.05 * tot_range)
                if all([np.isfinite(val) for val in xlim]):
                    plt.xlim(xlim)

    if ylim != 'default':
        if ylim is not None:
            plt.ylim(ylim)
        else:
            if plot_raw_data:
                ylim = stats.weighted_quantile(df[yvar].values, df[wvar].values, [0.005, 0.995])
                plt.ylim(ylim)
            else:
                bin_min = np.amin(by_bin[yvar].values)
                bin_max = np.amax(by_bin[yvar].values)
                tot_range = bin_max - bin_min
                ylim = (bin_min - 0.1 * tot_range, bin_max + 0.1 * tot_range)
                if all([np.isfinite(val) for val in ylim]):
                    plt.ylim(ylim)

    plt.tight_layout()
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)

    plt.close(fig)

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
