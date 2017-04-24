import matplotlib.pyplot as plt

def two_axis(df, var1, var2, filepath=None, 
                  label1=None, label2=None, 
                  loc1='upper left', loc2='upper right',
                  legend_font=10, label_font=12, normalize=False, 
                  color1='royalblue', color2='orangered', 
                  markevery=4):

    fig, ax1 = plt.subplots()

    if label1 is None:
        label1 = var1

    if label2 is None:
        label2 = var2

    leglabel1 = label1 + ' (left axis)'
    leglabel2 = label2 + ' (right axis)'

    df[var1].plot(ax=ax1, linewidth=2, label=leglabel1, color=color1)

    ax2 = ax1.twinx()
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
        ax2_ylim = tuple([
            df[var2].std() * (val - df[var1].mean()) / df[var1].std() 
            + df[var2].mean()
            for val in ax1_ylim
        ])

        ax2.set_ylim(ax2_ylim)

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close(fig)

    return None

def normalized(df, var_list, invert_list=None):
    
    if invert_list is None:
        invert_list = len(var_list) * [False]
    
    fig = plt.figure()
    
    for this_var, invert in zip(var_list, invert_list):
        
        x = df[this_var].values.copy()
        x -= np.mean(x)
        x /= np.std(x)
        
        if invert:
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
