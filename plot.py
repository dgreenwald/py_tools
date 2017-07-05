import matplotlib.pyplot as plt
import numpy as np

def two_axis(df_in, var1, var2, filepath=None, 
                  label1=None, label2=None, 
                  loc1='upper left', loc2='upper right',
                  legend_font=10, label_font=12, normalize=False, 
                  color1='#1f77b4', color2='#ff7f0e', 
                  flip1=False, flip2=False,
                  markevery=4):

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

def clean(df_in, var_list):

    df = df_in[var_list].replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def hist(df_in, var, label=None, xtitle=None, weight_var=None, 
         bins=None, ylim=None, filepath=None):

    df = clean(df_in, [var, weight_var])

    if var not in df or len(df) == 0:
        return False

    if weight_var is not None:
        w = df[weight_var].values
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

def double_hist(df_in0, df_in1, label0, label1, var, bins=None, 
                weight_var=None, filepath=None, xtitle=None, ylim=None):

    df0 = clean(df_in0, [var, weight_var])
    df1 = clean(df_in1, [var, weight_var])

    if var not in df0 or var not in df1:
        return False

    if len(df0) == 0 or len(df1) == 0:
        return False

    if weight_var is not None:
        w0 = df0[weight_var].values
    else:
        w0 = np.ones(len(df0))

    if weight_var in df1:
        w1 = df1[weight_var].values
    else:
        w1 = np.ones(len(df1))

    fig = plt.figure()
    plt.hist(df0[var].values, normed=True, bins=bins, alpha=0.5,
             weights=w0, label=str(label0))
    plt.hist(df1[var].values, normed=True, bins=bins, alpha=0.5,
             weights=w1, label=str(label1))
    plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle)
    if ylim is not None:
        plt.ylim((ylim))

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    return True
