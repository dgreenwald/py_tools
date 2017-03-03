import matplotlib.pyplot as plt

def two_axis_plot(df, var1, var2, filepath, label1=None, label2=None, 
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

    ax1.legend(loc='upper left', fontsize=legend_font)
    ax2.legend(loc='upper right', fontsize=legend_font)

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

    plt.savefig(filepath)
    plt.close(fig)

    return None
