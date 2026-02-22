import numpy as np
import pandas as pd

# import pickle
# import pdb
import os
from zipfile import ZipFile


def cat(num):
    """Return a list of integers from 1 to num inclusive.

    Parameters
    ----------
    num : int
        Upper bound of the range (inclusive).

    Returns
    -------
    list
        List of integers ``[1, 2, ..., num]``.
    """
    return list(range(1, num + 1))


def to_float(df, var):
    """Convert a DataFrame column to float64 using pd.to_numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to convert.
    var : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        DataFrame with column ``var`` cast to ``float64``; non-numeric
        values are coerced to ``NaN``.
    """
    df[var] = pd.to_numeric(df[var], errors="coerce").astype(np.float64)
    return df


def load_data(yr, q, dataset, **kwargs):
    """Load acquisition data for a given year/quarter, dispatching by dataset.

    Parameters
    ----------
    yr : int
        Origination year.
    q : int
        Origination quarter (1–4).
    dataset : str
        Data source to load; ``'fannie'`` or ``'freddie'``.
    **kwargs
        Additional keyword arguments forwarded to the underlying loader.

    Returns
    -------
    pandas.DataFrame or None
        Loaded acquisition data with standard numeric columns coerced to
        numeric types, or ``None`` if the requested file does not exist or
        an invalid ``dataset`` is supplied.
    """
    if dataset == "fannie":
        df = load_fannie(yr, q, **kwargs)
    elif dataset == "freddie":
        df = load_freddie(yr, q, **kwargs)
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


# def load_fannie(yr, q, use_pickle=True, reimport=False, **kwargs):
def load_fannie(yr, q, **kwargs):
    """Load a Fannie Mae acquisition text file for a given year and quarter.

    Reads the pipe-delimited acquisition file located at
    ``/nobackup1/dlg/fannie/Acquisition_{yr}Q{q}.txt``.

    Parameters
    ----------
    yr : int
        Origination year.
    q : int
        Origination quarter (1–4).
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_table``.

    Returns
    -------
    pandas.DataFrame or None
        Acquisition data with 25 standard columns, or ``None`` if the
        file does not exist.
    """

    data_dir = "/nobackup1/dlg/fannie/"
    # pkl_file = data_dir + 'Acquisition_{0}Q{1}.pkl'.format(yr, q)

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

        filepath = data_dir + "Acquisition_{0}Q{1}.txt".format(yr, q)
        compression = None

        if os.path.isfile(filepath):
            df = pd.read_table(
                filepath, sep="|", compression=compression, names=acq_names, **kwargs
            )
            # if use_pickle:
            # df.to_pickle(pkl_file)
            return df
        else:
            return None

    # else:
    # return pd.read_pickle(pkl_file)


def load_freddie(yr, q, **kwargs):
    """Load a Freddie Mac acquisition zip file for a given year and quarter.

    Reads the pipe-delimited acquisition file from the zip archive located
    at ``/nobackup1/dlg/freddie/data/historical_data1_Q{q}{yr}.zip``.

    Parameters
    ----------
    yr : int
        Origination year.
    q : int
        Origination quarter (1–4).
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_table``.

    Returns
    -------
    pandas.DataFrame or None
        Acquisition data with 25 standard columns, or ``None`` if the
        zip file does not exist.
    """

    data_dir = "/nobackup1/dlg/freddie/data"

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

    filename = "historical_data1_Q{0}{1}".format(q, yr)
    zipname = data_dir + "/" + filename + ".zip"
    txtname = filename + ".txt"

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_table(fid, sep="|", names=col_names, index_col=False, **kwargs)

            return df
    else:
        return None


def load_hmda_raw(yr, nrows=None, usecols=None):
    """Load raw HMDA data from a fixed-width or CSV file.

    Reads the Home Mortgage Disclosure Act LAR file for the given year
    from ``/nobackup1/dlg/hmda/ultimate/``.  Files before 2004 and from
    2004–2013 use fixed-width format; files from 2014 onward use CSV
    format.  All pre-2014 paths expect a ``.zip?download=true`` suffix.

    Parameters
    ----------
    yr : int
        HMDA filing year.
    nrows : int or None, optional
        Maximum number of rows to read.  Reads all rows when ``None``.
    usecols : list or None, optional
        Subset of column names to return.  Returns all columns when ``None``.

    Returns
    -------
    pandas.DataFrame
        Raw HMDA LAR records for the requested year.
    """

    data_dir = "/nobackup1/dlg/hmda/ultimate/"

    # Filename
    if yr == 2001:
        filename = "HMS.U2001.LARS.PUBLIC.DATA"
    elif yr == 2004:
        filename = "u2004lar.public.dat"
    elif yr in [2005, 2006]:
        filename = "LARS.ULTIMATE.{}.DAT".format(yr)
    elif yr in [2007, 2008]:
        filename = "lars.ultimate.{}.dat".format(yr)
    elif yr == 2009:
        filename = "2009_Ultimate_PUBLIC_LAR.dat"
    elif yr > 2009 and yr <= 2013:
        filename = "Lars.ultimate.{0}.dat".format(yr)
    elif yr > 2013:
        filename = "{0}HMDALAR - National.zip".format(yr)
    else:
        filename = "HMS.U{}.LARS".format(yr)

    filepath = data_dir + filename
    if yr < 2014:
        filepath += ".zip?download=true"

    if yr < 2004:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            4,
            2,
            3,
            7,
            1,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_race",
            "co_app_race",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "seq_num",
        ]

        df = pd.read_fwf(
            filepath,
            widths=widths,
            names=names,
            usecols=usecols,
            nrows=nrows,
            compression="zip",
        )

    elif yr < 2014:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            5,
            2,
            3,
            7,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "prop_type",
            "preapprovals",
            "app_ethnicity",
            "co_app_ethnicity",
            "app_race_1",
            "app_race_2",
            "app_race_3",
            "app_race_4",
            "app_race_5",
            "co_app_race_1",
            "co_app_race_2",
            "co_app_race_3",
            "co_app_race_4",
            "co_app_race_5",
            "rate_spread",
            "hoepa_status",
            "lien_status",
            "seq_num",
        ]

        df = pd.read_fwf(
            filepath,
            widths=widths,
            names=names,
            usecols=usecols,
            nrows=nrows,
            compression="zip",
        )

    else:
        # widths=[
        # 4, 10, 1, 1, 1,
        # 1, 1, 5, 1, 1,
        # 5, 2, 3, 7, 1,
        # 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1,
        # 1, 1, 1, 4, 1,
        # 1, 1, 1, 5, 1,
        # 1, 1, 7, 8, 6,
        # 8, 6, 8, 8, 1,
        # ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "prop_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "preapprovals",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_ethnicity",
            "co_app_ethnicity",
            "app_race_1",
            "app_race_2",
            "app_race_3",
            "app_race_4",
            "app_race_5",
            "co_app_race_1",
            "co_app_race_2",
            "co_app_race_3",
            "co_app_race_4",
            "co_app_race_5",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "rate_spread",
            "hoepa_status",
            "lien_status",
            "edit_status",
            "seq_num",
            "pop",
            "minority_pop_share",
            "median_family_inc",
            "tract_to_msa_inc",
            "n_units",
            "n_1to4_units",
            "app_date",
        ]

        df = pd.read_csv(
            filepath, names=names, usecols=usecols, nrows=nrows, compression="zip"
        )

    return df


def store_hmda(yr, nrows=None, usecols=None, reimport=False, chunksize=500000):
    """Read raw HMDA data in chunks and persist it to an HDF5 store.

    Reads the HMDA LAR file for the given year in chunks, drops
    demographic and identifier columns, coerces remaining columns to
    ``float64``, and appends each chunk to the key ``hmda_{yr}`` in the
    HDF5 store at ``/nobackup1/dlg/hmda/save/hmda.hd5``.

    Parameters
    ----------
    yr : int
        HMDA filing year.
    nrows : int or None, optional
        Maximum number of rows to read from the source file.
    usecols : list or None, optional
        Subset of column names to read from the source file.
    reimport : bool, optional
        Reserved for future use; currently unused.
    chunksize : int, optional
        Number of rows per chunk passed to the pandas reader.

    Returns
    -------
    None
    """

    data_dir = "/nobackup1/dlg/hmda/ultimate/"
    save_dir = "/nobackup1/dlg/hmda/save/"

    store_file = save_dir + "hmda.hd5"
    key = "hmda_{}".format(yr)
    store = pd.HDFStore(store_file)

    # Filename
    if yr == 2001:
        filename = "HMS.U2001.LARS.PUBLIC.DATA"
    elif yr == 2004:
        filename = "u2004lar.public.dat"
    elif yr in [2005, 2006]:
        filename = "LARS.ULTIMATE.{}.DAT".format(yr)
    elif yr in [2007, 2008]:
        filename = "lars.ultimate.{}.dat".format(yr)
    elif yr == 2009:
        filename = "2009_Ultimate_PUBLIC_LAR.dat"
    elif yr > 2009 and yr <= 2013:
        filename = "Lars.ultimate.{0}.dat".format(yr)
    elif yr > 2013:
        filename = "{0}HMDALAR - National.zip".format(yr)
    else:
        filename = "HMS.U{}.LARS".format(yr)

    filepath = data_dir + filename
    if yr < 2014:
        filepath += ".zip?download=true"

    if yr < 2004:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            4,
            2,
            3,
            7,
            1,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_race",
            "co_app_race",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "seq_num",
        ]

        reader = pd.read_fwf(
            filepath,
            widths=widths,
            names=names,
            usecols=usecols,
            nrows=nrows,
            compression="zip",
            chunksize=chunksize,
        )

    elif yr < 2014:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            5,
            2,
            3,
            7,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "prop_type",
            "preapprovals",
            "app_ethnicity",
            "co_app_ethnicity",
            "app_race_1",
            "app_race_2",
            "app_race_3",
            "app_race_4",
            "app_race_5",
            "co_app_race_1",
            "co_app_race_2",
            "co_app_race_3",
            "co_app_race_4",
            "co_app_race_5",
            "rate_spread",
            "hoepa_status",
            "lien_status",
            "seq_num",
        ]

        reader = pd.read_fwf(
            filepath,
            widths=widths,
            names=names,
            usecols=usecols,
            nrows=nrows,
            compression="zip",
            chunksize=chunksize,
        )

    else:
        # widths=[
        # 4, 10, 1, 1, 1,
        # 1, 1, 5, 1, 1,
        # 5, 2, 3, 7, 1,
        # 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1,
        # 1, 1, 1, 4, 1,
        # 1, 1, 1, 5, 1,
        # 1, 1, 7, 8, 6,
        # 8, 6, 8, 8, 1,
        # ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "prop_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "preapprovals",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_ethnicity",
            "co_app_ethnicity",
            "app_race_1",
            "app_race_2",
            "app_race_3",
            "app_race_4",
            "app_race_5",
            "co_app_race_1",
            "co_app_race_2",
            "co_app_race_3",
            "co_app_race_4",
            "co_app_race_5",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "rate_spread",
            "hoepa_status",
            "lien_status",
            "edit_status",
            "seq_num",
            "pop",
            "minority_pop_share",
            "median_family_inc",
            "tract_to_msa_inc",
            "n_units",
            "n_1to4_units",
            "app_date",
        ]

        reader = pd.read_csv(
            filepath,
            names=names,
            usecols=usecols,
            nrows=nrows,
            compression="zip",
            chunksize=chunksize,
        )

    data_columns = [
        "loan_type",
        "loan_purp",
        "occupancy",
        "action_taken",
        "lien_status",
        "purchaser_type",
    ]

    obj_columns = []
    drop_columns = [
        "resp_id",
        "agency_code",
        "app_sex",
        "co_app_sex",
        "app_ethnicity",
        "co_app_ethnicity",
        "app_race",
        "co_app_race",
        "app_race_1",
        "app_race_2",
        "app_race_3",
        "app_race_4",
        "app_race_5",
        "co_app_race_1",
        "co_app_race_2",
        "co_app_race_3",
        "co_app_race_4",
        "co_app_race_5",
        "hoepa_status",
        "seq_num",
        "app_date",
    ]

    # cat_vars = ['']
    categories = {}
    # categories = {
    # 'loan_type' : cat(4),
    # 'prop_type' : cat(3),
    # 'loan_purp' : cat(3),
    # 'occupancy' : cat(3),
    # 'preapprovals' : cat(3),
    # 'action_taken' : cat(8),
    # # 'denial_reason_1' : cat(9),
    # # 'denial_reason_2' : cat(9),
    # # 'denial_reason_3' : cat(9),
    # 'edit_status' : list(range(5, 8)),
    # # 'state_code' : list(range(1, 100)),
    # 'purchaser_type' : list(range(10)),
    # 'lien_status' : list(range(5)),
    # }

    for ii, df in enumerate(reader):
        if ii == 0:
            print("\n\n\nBEFORE:\n")
            print(df.head())
            print(df.dtypes)

        for col in df.columns:
            if col in drop_columns:
                df.drop(col, axis=1, inplace=True)
            elif col in obj_columns:
                df[col] = df[col].astype("object")
            elif col in categories:
                df[col] = pd.Categorical(df[col], categories=categories[col])
            else:
                to_float(df, col)

        if ii == 0:
            print("\n\n\nAFTER:\n")
            print(df.head())
            print(df.dtypes)

        if ii == 0:
            store.append(key, df, append=False, data_columns=data_columns)
        else:
            store.append(key, df, data_columns=data_columns)

    # if chunksize is None or (nrows is not None and nrows <= chunksize):
    # df = pd.read_fwf(filepath, widths=widths, names=names,
    # usecols=usecols, nrows=nrows, compression='zip')
    # else:
    # df = pd.DataFrame(columns=usecols)
    # reader = pd.read_fwf(filepath, widths=widths, names=names,
    # usecols=usecols, nrows=nrows, compression='zip',
    # chunksize=chunksize)

    # for df_in in reader:
    # df = df.append(df_in, ignore_index=True)

    return None


def load_hmda(yr, query=None, columns=None):
    """Load HMDA data for a given year from the HDF5 store.

    Opens the HDF5 store at ``/nobackup1/dlg/hmda/save/hmda.hd5`` and
    reads the key ``hmda_{yr}``, optionally filtering rows and columns.

    Parameters
    ----------
    yr : int
        HMDA filing year.
    query : str or None, optional
        PyTables query string used to filter rows (e.g.
        ``"loan_type == 1"``).  Returns all rows when ``None``.
    columns : list or None, optional
        List of column names to return.  Returns all columns when ``None``.

    Returns
    -------
    pandas.DataFrame
        HMDA records for the requested year.
    """

    save_dir = "/nobackup1/dlg/hmda/save/"

    store_file = save_dir + "hmda.hd5"
    key = "hmda_{}".format(yr)
    store = pd.HDFStore(store_file)

    store.open()
    if query is None and columns is None:
        df = pd.read_hdf(store, key)
    else:
        df = store.select(key, query, columns=columns)
    store.close()

    return df


def load_black_knight(yr, suffix=None):
    """Load Black Knight data for a given year from an HDF5 store.

    .. note::
        This function is a placeholder and is not yet implemented.
        It immediately raises ``Exception`` on every call.

    Parameters
    ----------
    yr : int
        Data year used to construct the HDF5 key ``yr_{yr}``.
    suffix : str or None, optional
        Optional suffix appended to the store filename
        (``black_knight_{suffix}.hd5``).  A leading underscore is added
        automatically if absent.

    Returns
    -------
    None
        This function never returns normally.

    Raises
    ------
    Exception
        Always raised; function body is not implemented.
    """

    raise Exception

    if suffix is None:
        suffix = ""
    elif suffix[0] != "_":
        suffix = "_" + suffix

    save_dir = "/nobackup1/dlg/black_knight/hd5/"
    store_file = save_dir + "black_knight{}.hd5".format(suffix)
    key = "yr_{}".format(yr)

    return pd.read_hdf(store_file, key)


# def store_black_knight(yr)

# data_dir = '/nfs/sloanlab001/data/BlackKnight/'
# hd5_dir = '/nobackup1/dlg/black_knight/hd5/'

# store_file = hd5_dir + 'black_knight.hd5'
