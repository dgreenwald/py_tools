import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import os

default_dir = "/data/fannie/"
# data_dir = '/data/fannie/'
DATASET_NAME = "fannie"
DESCRIPTION = "Fannie Mae mortgage acquisition/performance dataset loader."


def load(year, q, reimport=False, **kwargs):
    """Load Fannie Mae acquisition data for a single year/quarter.

    Reads a pipe-delimited text file (``Acquisition_<year>Q<q>.txt``) and
    caches the result as a pickle.  If a cached pickle already exists and
    *reimport* is ``False``, the pickle is returned directly.

    Parameters
    ----------
    year : int
        Four-digit origination year of the Fannie Mae data file.
    q : int
        Quarter of the data file (1–4).
    reimport : bool, optional
        If ``True``, re-read the raw text file even when a cached pickle
        exists.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`pandas.read_table`.  ``data_dir`` may be passed here to
        override the default data directory.

    Returns
    -------
    pandas.DataFrame or None
        Acquisition DataFrame for the requested quarter, or ``None`` if the
        source text file does not exist.
    """
    data_dir = kwargs.pop("data_dir", default_dir)

    pkl_file = data_dir + "/Acquisition_{0}Q{1}.pkl".format(year, q)

    if os.path.isfile(pkl_file) and not reimport:
        return pd.read_pickle(pkl_file, **kwargs)

    acq_names = [
        "loan_id",
        "channel",
        "seller_name",
        "orig_int_rate",
        "orig_upb",
        "orig_term",
        "orig_date",
        "first_pay_date",
        "orig_ltv",
        "orig_cltv",
        "n_borrowers",
        "orig_dti",
        "credit_score",
        "first_time_flag",
        "loan_purpose",
        "prop_type",
        "n_units",
        "occ_status",
        "prop_state",
        "zip3",
        "mi_pct",
        "prod_type",
        "co_borr_credit_score",
        "mi_type",
        "reloc",
    ]

    filepath = data_dir + "/Acquisition_{0}Q{1}.txt".format(year, q)
    if not os.path.isfile(filepath):
        return None

    df = pd.read_table(filepath, sep="|", names=acq_names, **kwargs)
    for date_var in ["orig_date", "first_pay_date"]:
        df[date_var] = pd.to_datetime(df[date_var])

    df["orig_upb"] = df["orig_upb"].astype(np.float64) / 1000.0

    df.to_pickle(pkl_file)

    return df


def hist(
    df,
    var_list,
    yr,
    q,
    base_title=None,
    titlestr=None,
    filestr=None,
    prefix="fannie",
    print_nobs=False,
    filetype="png",
    out_dir=None,
    save_fig_pkl=True,
    save_hist_pkl=True,
    print_title=True,
    print_xlabel=True,
    smallfont=16,
    vertical_line=True,
    var_titles=None,
    plot_titles=None,
    **kwargs,
):
    """Plot weighted histograms of loan characteristics for a Fannie Mae quarter.

    For each variable in *var_list* a histogram is produced weighted by
    ``orig_upb``.  When *out_dir* is provided the figures are saved to disk;
    otherwise they are shown interactively.

    Parameters
    ----------
    df : pandas.DataFrame
        Fannie Mae acquisition DataFrame containing the variables in
        *var_list* and an ``orig_upb`` column used for weighting.
    var_list : list of str
        Column names to plot (e.g. ``['orig_dti', 'orig_ltv']``).
    yr : int
        Year label used in the output file name and plot title.
    q : int
        Quarter label (1–4) used in the output file name and plot title.
    base_title : str, optional
        Base string prepended to every plot title.  Defaults to
        ``'Fannie Mae'``.
    titlestr : str, optional
        Additional string appended to each plot title after the variable
        name.
    filestr : str, optional
        Extra string inserted into the output file name.
    prefix : str, optional
        Prefix for output file names.
    print_nobs : bool, optional
        If ``True``, append the number of observations to the plot title.
    filetype : str, optional
        File extension / format for saved figures (e.g. ``'png'``,
        ``'pdf'``).
    out_dir : str, optional
        Directory where figures are saved.  If ``None``, figures are
        displayed interactively.
    save_fig_pkl : bool, optional
        If ``True``, pickle the :class:`matplotlib.figure.Figure` object
        alongside the image file.
    save_hist_pkl : bool, optional
        If ``True``, pickle the histogram output tuple alongside the image
        file.
    print_title : bool, optional
        If ``True``, render the title on each figure.
    print_xlabel : bool, optional
        If ``True``, render the x-axis label on each figure.
    smallfont : int, optional
        Global font size passed to :func:`matplotlib.rcParams.update`.
    vertical_line : bool, optional
        If ``True``, draw a vertical reference line on DTI histograms at
        46 %.
    var_titles : dict, optional
        Mapping from column name to a short display label used in titles
        and fallback x-axis labels.
    plot_titles : dict, optional
        Mapping from column name to the x-axis label string.
    **kwargs
        Additional keyword arguments forwarded to :func:`matplotlib.pyplot.hist`.

    Returns
    -------
    None
    """

    matplotlib.rcParams.update({"font.size": smallfont})

    if base_title is None:
        base_title = "Fannie Mae"
    base_title += ": "

    if var_titles is None:
        var_titles = {
            "orig_dti": "PTI Ratio",
            "orig_ltv": "LTV Ratio",
            "orig_cltv": "CLTV Ratio",
            "credit_score": "Credit Score",
            "orig_upb": "Orig. Balance",
        }

    if plot_titles is None:
        plot_titles = {
            "orig_dti": "PTI Ratio (%)",
            "orig_ltv": "LTV Ratio (%)",
            "orig_cltv": "CLTV Ratio (%)",
            "credit_score": "Credit Score",
            "orig_upb": "Orig. Balance ($k)",
        }

    for var in var_list:
        filename = prefix + "_hist_" + var
        if filestr is not None:
            filename += "_" + filestr
        filename += "_{0}_Q{1}".format(yr, q)

        title = base_title
        title += var_titles.get(var, var)

        if titlestr is not None:
            title += ", " + titlestr

        if var == "orig_dti":
            bins = np.linspace(0.0, 70.0, 71)
        elif var in ["orig_ltv", "orig_cltv"]:
            # bins = np.linspace(50.0, 100.0, 51)
            bins = np.linspace(50.0, 110.0, 61)
        elif var in ["credit_score"]:
            bins = np.linspace(500.0, 800.0, 31)
        elif var in ["orig_upb"]:
            bins = np.linspace(0.0, 800.0, 41)
        else:
            bins = None

        ix_hist = np.ones(len(df), dtype=bool)
        if not np.any(ix_hist):
            return None
        ix_hist = np.logical_and(ix_hist, pd.notnull(df[var]))
        if not np.any(ix_hist):
            return None
        ix_hist = np.logical_and(ix_hist, pd.notnull(df["orig_upb"]))
        if not np.any(ix_hist):
            return None

        if print_nobs:
            title += " (nobs = {})".format(np.sum(ix_hist))

        fig = plt.figure()
        hist_out = plt.hist(
            x=df.loc[ix_hist, var].values,
            weights=df.loc[ix_hist, "orig_upb"].values,
            bins=bins,
            normed=True,
            **kwargs,
        )

        if var == "orig_dti":
            if vertical_line:
                plt.axvline(x=46.0, linewidth=2, color="r")
            # if print_xlabel:
            # plt.xlabel('PTI Ratio (%)')
            plt.ylim((0.0, 0.1))
            nbins = 5
        elif var == "orig_ltv":
            # if print_xlabel:
            # plt.xlabel('LTV Ratio (%)')
            plt.ylim((0.0, 0.4))
            nbins = 4
        elif var == "orig_cltv":
            # if print_xlabel:
            # plt.xlabel('CLTV Ratio (%)')
            plt.ylim((0.0, 0.4))
            nbins = 4
        else:
            # plt.xlabel(var_titles.get(var, var))
            nbins = 5

        if print_xlabel:
            plt.xlabel(plot_titles.get(var, var_titles.get(var, var)))

        plt.locator_params(axis="y", nbins=nbins)

        if print_title:
            plt.title(title)

        if out_dir is None:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig("{0}{1}.{2}".format(out_dir, filename, filetype))

            if save_hist_pkl:
                pickle.dump(
                    hist_out, open("{0}{1}_hist.pkl".format(out_dir, filename), "wb")
                )

            if save_fig_pkl:
                pickle.dump(fig, open("{0}{1}_fig.pkl".format(out_dir, filename), "wb"))

        plt.close(fig)

    return None
