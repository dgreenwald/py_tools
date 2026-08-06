import itertools

import numpy as np
import pandas as pd

# import pickle
# import pdb
import os
from zipfile import ZipFile

from py_tools import in_out

DEFAULT_FANNIE_DIR = os.environ["HOME"] + "/data/fannie/"
DEFAULT_FREDDIE_DIR = os.environ["HOME"] + "/data/freddie/"
DATASET_NAME = "gse"
DESCRIPTION = "GSE mortgage datasets loader (Fannie/Freddie)."


def cat(num):
    """Return a list of integers from 1 to num (inclusive).

    Parameters
    ----------
    num : int
        Upper bound of the range.

    Returns
    -------
    list
        List of integers [1, 2, ..., num].
    """
    return list(range(1, num + 1))


def to_float(df, var):
    """Convert a DataFrame column to float64 if its dtype is object.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to convert.
    var : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the specified column cast to float64 when applicable.
    """
    if df[var].dtype == "object":
        df[var] = pd.to_numeric(df[var], errors="coerce").astype(np.float64)
    return df


def load_all(dataset, **kwargs):
    """Load all quarters from 2000 to 2018 for a GSE dataset and concatenate.

    Parameters
    ----------
    dataset : str
        Name of the GSE dataset to load.  Must be one of
        ``'fannie_acquisition'``, ``'freddie_acquisition'``, or
        ``'freddie_performance'``.
    **kwargs
        Additional keyword arguments forwarded to :func:`load`.

    Returns
    -------
    pandas.DataFrame
        Concatenated data across all year-quarter combinations from 2000 Q1
        through 2018 Q4.
    """
    df_list = []
    for year, q in itertools.product(range(2000, 2019), range(1, 5)):
        print("Loading {0:d} Q{1:d}".format(year, q))

        df_t = load(year, q, dataset, **kwargs)
        df_list.append(df_t)

    df = pd.concat(df_list, axis=0)
    return df


def load(
    year,
    q,
    dataset,
    reimport=False,
    data_dir=None,
    columns=None,
    compression="GZIP",
    **kwargs,
):
    """Load a single GSE quarter, reading from a parquet cache or re-importing.

    If a cached parquet file exists and ``reimport`` is ``False``, the data are
    read directly from that file.  Otherwise the raw source file is imported,
    numeric and date columns are coerced, and the result is written to parquet.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    dataset : str
        Dataset to load.  Must be one of ``'fannie_acquisition'``,
        ``'freddie_acquisition'``, or ``'freddie_performance'``.
    reimport : bool, optional
        When ``True``, ignore any existing parquet cache and re-import from the
        source file.  Default is ``False``.
    data_dir : str, optional
        Path to the directory containing the raw source files.  Defaults to the
        dataset-specific ``DEFAULT_*_DIR`` constant when ``None``.
    columns : list of str, optional
        Subset of columns to return when reading from the parquet cache.
        Default is ``None`` (all columns).
    compression : str, optional
        Compression codec used when writing the parquet file.  Default is
        ``'GZIP'``.
    **kwargs
        Additional keyword arguments forwarded to the underlying import
        function.

    Returns
    -------
    pandas.DataFrame or None
        Loaded data for the requested quarter, or ``None`` if the source file
        does not exist.

    Raises
    ------
    Exception
        When ``dataset`` is not one of the recognised dataset names.
    """
    default_data_dirs = {
        "fannie_acquisition": DEFAULT_FANNIE_DIR,
        "freddie_acquisition": DEFAULT_FREDDIE_DIR,
        "freddie_performance": DEFAULT_FREDDIE_DIR,
    }

    assert dataset in default_data_dirs

    # Set data directory if not specified
    if data_dir is None:
        data_dir = default_data_dirs[dataset]

    # Set up storage directory
    parquet_dir = data_dir + "storage/"
    in_out.make_dir(parquet_dir)

    # Load parquet file directly
    parquet_file = "{0}{1}_{2:d}Q{3:d}.parquet".format(parquet_dir, dataset, year, q)
    if os.path.exists(parquet_file) and (not reimport):
        return pd.read_parquet(parquet_file, columns=columns)

    # Otherwise, reimport from txt file
    if dataset == "fannie_acquisition":
        df = import_fannie_acquisition(year, q, data_dir=data_dir, **kwargs)
    elif dataset == "freddie_acquisition":
        df = import_freddie_acquisition(year, q, data_dir=data_dir, **kwargs)
    elif dataset == "freddie_performance":
        df = import_freddie_performance(year, q, data_dir=data_dir, **kwargs)
    else:
        raise Exception

    # Convert variables from string to numeric
    if df is not None:
        numerical_list = [
            "orig_int_rate",
            "orig_upb",
            "orig_term",
            "orig_ltv",
            "orig_cltv",
            "orig_dti",
            "credit_score",
        ]
        for col in df.columns:
            if col in numerical_list:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        date_list = ["orig_date", "first_pay_date"]
        for col in df.columns:
            if col in date_list:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Save as parquet
        df.to_parquet(parquet_file, compression=compression)

    return df


def load_data(year, q, dataset, **kwargs):
    """Legacy dispatcher that loads Fannie Mae or Freddie Mac acquisition data.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    dataset : str
        Dataset identifier.  Use ``'fannie'`` for Fannie Mae or ``'freddie'``
        for Freddie Mac acquisition data.
    **kwargs
        Additional keyword arguments forwarded to :func:`load_fannie` or
        :func:`load_freddie`.

    Returns
    -------
    pandas.DataFrame or None
        Loaded acquisition data with numeric columns coerced, or ``None`` if
        the source file does not exist or ``dataset`` is unrecognised.
    """
    if dataset == "fannie":
        df = load_fannie(year, q, **kwargs)
    elif dataset == "freddie":
        df = load_freddie(year, q, **kwargs)
    else:
        print("Invalid data set")
        return None

    if df is not None:
        numerical_list = [
            "orig_int_rate",
            "orig_upb",
            "orig_term",
            "orig_ltv",
            "orig_cltv",
            "orig_dti",
            "credit_score",
        ]
        for col in df.columns:
            if col in numerical_list:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def import_fannie_acquisition(year, q, data_dir=DEFAULT_FANNIE_DIR, **kwargs):
    """Import a Fannie Mae acquisition text file for a given year and quarter.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    data_dir : str, optional
        Path to the directory containing Fannie Mae acquisition text files.
        Defaults to ``DEFAULT_FANNIE_DIR``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Parsed acquisition data, or ``None`` if the expected file does not
        exist.
    """
    col_names = [
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

    filepath = data_dir + "Acquisition_{0}Q{1}.txt".format(year, q)
    compression = None

    if os.path.isfile(filepath):
        df = pd.read_csv(
            filepath, sep="|", compression=compression, names=col_names, **kwargs
        )
        return df
    else:
        return None


def import_freddie_acquisition(year, q, data_dir=DEFAULT_FREDDIE_DIR, **kwargs):
    """Import a Freddie Mac acquisition zip file for a given year and quarter.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    data_dir : str, optional
        Path to the directory containing Freddie Mac acquisition zip files.
        Defaults to ``DEFAULT_FREDDIE_DIR``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Parsed acquisition data, or ``None`` if the expected zip file does not
        exist.
    """
    col_names = [
        "credit_score",
        "first_pay_date",
        "first_time_flag",
        "maturity_date",
        "msa",
        "mi_pct",
        "n_units",
        "occ_status",
        "orig_cltv",
        "orig_dti",
        "orig_upb",
        "orig_ltv",
        "orig_int_rate",
        "channel",
        "prepay_pen_flag",
        "prod_type",
        "prop_state",
        "prop_type",
        "zip3",
        "loan_id",
        "loan_purpose",
        "orig_term",
        "n_borrowers",
        "seller_name",
        "servicer_name",
    ]

    filename = "historical_data1_Q{0}{1}".format(q, year)
    zipname = data_dir + "/" + filename + ".zip"
    txtname = filename + ".txt"

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep="|", names=col_names, index_col=False, **kwargs)
            return df
    else:
        return None


def import_freddie_performance(year, q, data_dir=DEFAULT_FREDDIE_DIR, **kwargs):
    """Import a Freddie Mac performance zip file for a given year and quarter.

    Reads the pipe-delimited performance file, coerces data types, recodes the
    delinquency field to handle REO codes, and splits ``net_sales_proceeds``
    into a numeric amount column and a categorical code column.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    data_dir : str, optional
        Path to the directory containing Freddie Mac performance zip files.
        Defaults to ``DEFAULT_FREDDIE_DIR``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Parsed and processed performance data, or ``None`` if the expected zip
        file does not exist.
    """
    col_names = [
        "loan_id",
        "asof_date",
        "current_upb",
        "delinq",
        "loan_age",
        "months_left",
        "repurchase",
        "modification",
        "zero_balance",
        "zero_balance_date",
        "current_rate",
        "current_deferred_upb",
        "last_due_date",
        "mi_recoveries",
        "net_sales_proceeds",
        "non_mi_recoveries",
        "expenses",
        "legal_costs",
        "maintenance_costs",
        "taxes_and_insurance",
        "misc_expenses",
        "actual_loss",
        "modification_cost",
        "step_modification_flag",
        "deferred_pay_mod",
        "estimated_ltv",
    ]

    filename = "historical_data1_time_Q{0}{1}".format(q, year)
    zipname = "{0}/historical_data1_Q{1}{2}.zip".format(data_dir, q, year)
    txtname = filename + ".txt"

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep="|", names=col_names, index_col=False, **kwargs)

        # Adjust object data types
        df["loan_id"] = df["loan_id"].astype(str)

        for var in ["repurchase", "modification", "step_modification_flag"]:
            df[var] = df[var].astype("category")

        # Update delinquency to account for REOs
        df["reo"] = df["delinq"] == "R"
        df.loc[df["reo"], "delinq"] = -1
        df["delinq"] = pd.to_numeric(df["delinq"], errors="coerce")
        df.loc[pd.isnull(df["delinq"]), "delinq"] = -2
        df["delinq"] = df["delinq"].astype(np.int64)

        # Update net_sales_proceeds codes
        df["net_sales_proceeds_amt"] = pd.to_numeric(
            df["net_sales_proceeds"], errors="coerce"
        )
        ix = pd.isnull(df["net_sales_proceeds_amt"])
        df.loc[ix, "net_sales_proceeds_code"] = df.loc[ix, "net_sales_proceeds"]
        df["net_sales_proceeds_code"] = df["net_sales_proceeds_code"].astype("category")
        df = df.drop(columns=["net_sales_proceeds"])

        return df

    else:
        return None


# def load_fannie(year, q, use_pickle=True, reimport=False, **kwargs):
def load_fannie(year, q, reimport=False, load_parquet=False, **kwargs):
    """Load Fannie Mae acquisition data with optional parquet cache.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    reimport : bool, optional
        Reserved for future use; currently ignored.  Default is ``False``.
    load_parquet : bool, optional
        When ``True``, attempt to read from an existing parquet file before
        falling back to the raw text file.  Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Acquisition data for the requested quarter, or ``None`` if the source
        file does not exist.
    """
    data_dir = "/nobackup1/dlg/fannie/"
    parquet_dir = data_dir + "/storage/"
    in_out.make_dir(parquet_dir)

    parquet_file = parquet_dir + "acquisition_{0:d}q{1:d}.parquet".format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    # if (not use_pickle) or reimport or (not os.path.exists(pkl_file)):
    if True:
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

        filepath = data_dir + "Acquisition_{0}Q{1}.txt".format(year, q)
        compression = None

        if os.path.isfile(filepath):
            df = pd.read_csv(
                filepath, sep="|", compression=compression, names=acq_names, **kwargs
            )
            # if use_pickle:
            # df.to_pickle(pkl_file)
            return df
        else:
            return None

    # else:
    # return pd.read_pickle(pkl_file)


def load_freddie(
    year, q, load_parquet=True, save_parquet=True, overwrite_parquet=False, **kwargs
):
    """Load Freddie Mac acquisition data with optional parquet cache.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    load_parquet : bool, optional
        When ``True``, attempt to read from an existing parquet file before
        importing the raw zip file.  Default is ``True``.
    save_parquet : bool, optional
        When ``True``, write the imported data to a parquet file for future
        use.  Default is ``True``.
    overwrite_parquet : bool, optional
        When ``True``, overwrite an existing parquet file even if it already
        exists.  Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Acquisition data for the requested quarter, or ``None`` if the source
        zip file does not exist.
    """
    data_dir = "/nobackup1/dlg/freddie/data"
    parquet_dir = data_dir + "/storage/"

    parquet_file = parquet_dir + "acquisition_{0:d}q{1:d}.parquet".format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    col_names = [
        "credit_score",
        "first_pay_date",
        "first_time_flag",
        "maturity_date",
        "msa",
        "mi_pct",
        "n_units",
        "occ_status",
        "orig_cltv",
        "orig_dti",
        "orig_upb",
        "orig_ltv",
        "orig_int_rate",
        "channel",
        "prepay_pen_flag",
        "prod_type",
        "prop_state",
        "prop_type",
        "zip3",
        "loan_id",
        "loan_purpose",
        "orig_term",
        "n_borrowers",
        "seller_name",
        "servicer_name",
    ]

    filename = "historical_data1_Q{0}{1}".format(q, year)
    zipname = data_dir + "/" + filename + ".zip"
    txtname = filename + ".txt"

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep="|", names=col_names, index_col=False, **kwargs)

            if save_parquet:
                if not os.path.exists(parquet_file) or overwrite_parquet:
                    print("Saving " + parquet_file)
                    df.to_parquet(parquet_file)

            return df
    else:
        return None


def load_freddie_performance(
    year, q, load_parquet=True, save_parquet=True, overwrite_parquet=False, **kwargs
):
    """Load Freddie Mac performance data with optional parquet cache.

    Parameters
    ----------
    year : int
        Four-digit calendar year of the origination quarter.
    q : int
        Quarter number (1–4).
    load_parquet : bool, optional
        When ``True``, attempt to read from an existing parquet file before
        importing the raw zip file.  Default is ``True``.
    save_parquet : bool, optional
        When ``True``, write the imported data to a parquet file for future
        use.  Default is ``True``.
    overwrite_parquet : bool, optional
        When ``True``, overwrite an existing parquet file even if it already
        exists.  Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame or None
        Performance data for the requested quarter, or ``None`` if the source
        zip file does not exist.
    """
    data_dir = "/nobackup1/dlg/freddie/data"
    parquet_dir = data_dir + "/storage/"

    parquet_file = parquet_dir + "perf_{0:d}q{1:d}.parquet".format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    col_names = [
        "loan_id",
        "asof_date",
        "current_upb",
        "delinq",
        "loan_age",
        "months_left",
        "repurchase",
        "modification",
        "zero_balance",
        "zero_balance_date",
        "current_rate",
        "current_deferred_upb",
        "last_due_date",
        "mi_recoveries",
        "net_sales_proceeds",
        "non_mi_recoveries",
        "expenses",
        "legal_costs",
        "maintenance_costs",
        "taxes_and_insurance",
        "misc_expenses",
        "actual_loss",
        "modification_cost",
        "step_modification_flag",
        "deferred_pay_mod",
        "estimated_ltv",
    ]

    filename = "historical_data1_time_Q{0}{1}".format(q, year)
    zipname = "{0}/historical_data1_Q{1}{2}.zip".format(data_dir, q, year)
    txtname = filename + ".txt"

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep="|", names=col_names, index_col=False, **kwargs)

        # Adjust object data types
        df["loan_id"] = df["loan_id"].astype(str)

        for var in ["repurchase", "modification", "step_modification_flag"]:
            df[var] = df[var].astype("category")

        # Update delinquency to account for REOs
        df["reo"] = df["delinq"] == "R"
        df.loc[df["reo"], "delinq"] = -1
        df["delinq"] = pd.to_numeric(df["delinq"], errors="coerce")
        df.loc[pd.isnull(df["delinq"]), "delinq"] = -2
        df["delinq"] = df["delinq"].astype(np.int64)

        # Update net_sales_proceeds codes
        df["net_sales_proceeds_amt"] = pd.to_numeric(
            df["net_sales_proceeds"], errors="coerce"
        )
        ix = pd.isnull(df["net_sales_proceeds_amt"])
        df.loc[ix, "net_sales_proceeds_code"] = df.loc[ix, "net_sales_proceeds"]
        df["net_sales_proceeds_code"] = df["net_sales_proceeds_code"].astype("category")
        df = df.drop(columns=["net_sales_proceeds"])

        if save_parquet:
            if not os.path.exists(parquet_file) or overwrite_parquet:
                print("Saving " + parquet_file)
                df.to_parquet(parquet_file)

        return df
    else:
        return None
