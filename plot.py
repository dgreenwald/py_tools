import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_tools.data import clean

def two_axis(df_in, var1, var2, filepath=None, 
                  label1=None, label2=None, 
                  loc1='upper left', loc2='upper right',
                  legend_font=10, label_font=12, normalize=False, 
                  color1='#1f77b4', color2='#ff7f0e', 
                  flip1=False, flip2=False,
                  markevery=4, legend=True):

    df = df_in[[var1, var2]].dropna()

    fig, ax1 = plt.subplots()

    if label1 is None:
        label1 = var1

    if label2 is None:
        label2 = var2

    leglabel1 = label1 + ' (left axis)'
    leglabel2 = label2 + ' (right axis)'

    if flip1:
        (-df[var1]).plot(ax=ax1, linewidth=2, label=('(-1) x ' + leglabel1), color=color1)
    else:
        df[var1].plot(ax=ax1, linewidth=2, label=leglabel1, color=color1)

    ax2 = ax1.twinx()
    if flip2:
        (-df[var2]).plot(ax=ax2, linestyle='-', linewidth=2, label=('(-1) x ' + leglabel2), 
                         color=color2, marker='o', fillstyle='none', markersize=5, 
                         mew=1.5, markevery=markevery)
    else:
        df[var2].plot(ax=ax2, linestyle='-', linewidth=2, label=leglabel2, color=color2,
                      marker='o', fillstyle='none', markersize=5, mew=1.5, markevery=markevery)

    if legend:
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

def hist(df_in, var, label=None, xtitle=None, wvar=None, 
         bins=None, ylim=None, filepath=None):

    if wvar is None:
        varlist = [var]
    else:
        varlist = [var, wvar]

    df = clean(df_in[varlist])

    if var not in df or len(df) == 0:
        return False

    if wvar is not None:
        w = df[wvar].values
    else:
        w = np.ones(len(df))

    fig = plt.figure()
    plt.hist(df[var].values, normed=True, bins=bins, alpha=0.5,
             weights=w, label=label)

    if xtitle is not None:
        plt.xlabel(xtitle)
    if ylim is not None:
        plt.ylim((ylim))
    if label is not None:
        plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    return True

def double_hist(df_in1, df_in2, label1='Var 1', label2='Var 2', var=None,
                var1=None, var2=None, bins=None, wvar=None, wvar1=None,
                wvar2=None, filepath=None, xtitle=None, ylim=None,
                legend_font=10, label_font=12):

    if var is not None:
        assert var1 is None and var2 is None
        var1 = var
        var2 = var

    if wvar is not None:
        assert wvar1 is None and wvar2 is None
        wvar1 = wvar
        wvar2 = wvar

    df1 = clean(df_in1[[var, wvar1]])
    df2 = clean(df_in2[[var, wvar2]])

    if var not in df1 or var not in df2:
        return False

    if len(df1) == 0 or len(df2) == 0:
        return False

    if wvar2 is not None:
        w1 = df1[wvar].values
    else:
        w1 = np.ones(len(df1))

    if wvar in df2:
        w2 = df2[wvar].values
    else:
        w2 = np.ones(len(df2))

    fig = plt.figure()
    matplotlib.rcParams.update({'font.size' : label_font})

    plt.hist(df1[var].values, normed=True, bins=bins, alpha=0.5,
             weights=w1, label=str(label1))
    plt.hist(df2[var].values, normed=True, bins=bins, alpha=0.5,
             weights=w2, label=str(label2))
    plt.legend(fontsize=legend_font)

    if xtitle is not None:
        plt.xlabel(xtitle, fontsize=label_font)
    if ylim is not None:
        plt.ylim((ylim))

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

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

def plot_series(df_in, var_names, directory, title=None, labels={},
                linestyles={}, markers={}, markevery=8, markersize=5, mew=2,
                fillstyle='none', fontsize=12, plot_type='pdf', ylabel=None):

    matplotlib.rcParams.update({'font.size' : fontsize})

    if title is None:
        title = '_'.join(var_names)

    fig = plt.figure()

    ix = np.any(pd.notnull(df_in[var_names]), axis=1)
    df = df_in.loc[ix, var_names].copy()

    for var in var_names:

        label = labels.get(var, var)
        linestyle = linestyles.get(var, '-')

        marker = markers.get(var, None)
        plt.plot(df.index, df[var], linewidth=2, linestyle=linestyle, label=label,
                 markevery=markevery, markersize=markersize, mew=mew)

    if len(var_names) > 1:
        plt.legend(fontsize=fontsize)

    plt.xlim(df.index[[0, -1]])

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig('{0}{1}.{2}'.format(directory, title, plot_type))

    plt.close(fig)

