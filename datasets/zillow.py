import os
import re

import pandas as pd

from . import config, misc

default_dir = config.base_dir() + "zillow/"
# data_dir = '/home/dan/data/irs/'
DATASET_NAME = "zillow"
DESCRIPTION = "Zillow housing dataset loader."


def load(geo, data_dir=default_dir, dataset="Zhvi_AllHomes", reimport=False):
    """Load Zillow home value data for a geography type.

    Reads the raw CSV for the given geography, reshapes from wide to long
    format, and merges a FIPS crosswalk (County) or state codes (State) as
    appropriate.  Results are cached as a pickle file so subsequent calls
    skip the CSV parse unless ``reimport`` is ``True``.

    Parameters
    ----------
    geo : str
        Geography level to load.  One of ``'State'``, ``'County'``,
        ``'Zip'``, or ``'Metro'``.  Case-insensitive; capitalised
        internally.
    data_dir : str, optional
        Path to the root Zillow data directory.  Defaults to the
        package-configured base directory.
    dataset : str, optional
        Zillow dataset name (file stem), e.g. ``'Zhvi_AllHomes'``.
    reimport : bool, optional
        When ``True``, re-read the CSV and overwrite any existing pickle
        cache.  Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame indexed by geography identifier and date.
    """
    geo = geo.capitalize()

    pattern = re.compile(r"\d\d\d\d-\d\d")

    pkl_file = data_dir + "pkl/{0}_{1}.pkl".format(geo, dataset)
    if reimport or (not os.path.exists(pkl_file)):
        df_wide = load_csv(data_dir=data_dir, dataset=dataset, geo=geo)
        value_vars = [col for col in df_wide.columns if pattern.match(col)]
        id_vars = [col for col in df_wide.columns if not pattern.match(col)]

        # if geo in ['State', 'Metro']:
        #     id_vars = ['RegionName']
        #     # value_vars = df_wide.columns[3:]
        # elif geo in ['Zip', 'County']:
        #     id_vars = df_wide.columns[:7]
        #     # value_vars = df_wide.columns[7:]

        df = pd.melt(df_wide, id_vars=id_vars, value_vars=value_vars)
        # df = df.rename(columns={
        #     col : col.lower() for col in df.columns
        #     })
        df = df.rename(columns={"variable": "date", "value": dataset})

        df["date"] = pd.to_datetime(df["date"])

        if geo == "County":
            cw = load_crosswalk(data_dir=data_dir)
            df = pd.merge(df, cw, left_on="RegionID", right_on="CountyRegionID_Zillow")
            df = df.set_index(["FIPS", "date"])
        elif geo == "State":
            state_codes = misc.load("state_codes")
            state_codes["state_name"] = state_codes["state_name"].str.title()
            df = pd.merge(df, state_codes, left_on="RegionName", right_on="state_name")
            df = df.drop(columns=["RegionName"])
            df = df.set_index(["state_abbr", "date"])
        elif geo == "Zip":
            df = df.rename(columns={"RegionName": "ZIP"})
            df = df.set_index(["ZIP", "date"])
        elif geo == "Metro":
            df = df.rename(columns={"RegionName": "metro"})
            df = df.set_index(["metro", "date"])

        df = df.sort_index()
        df.to_pickle(pkl_file)

    else:
        df = pd.read_pickle(pkl_file)

    return df


def load_county(data_dir=default_dir, dataset="Zhvi_AllHomes"):
    """Load Zillow county home value data and merge with FIPS crosswalk.

    Reads the county-level wide CSV, melts it to long format, converts the
    date column, and joins the Zillow county-FIPS crosswalk so the result
    is indexed by ``(FIPS, date)``.

    Parameters
    ----------
    data_dir : str, optional
        Path to the root Zillow data directory.  Defaults to the
        package-configured base directory.
    dataset : str, optional
        Zillow dataset name (file stem), e.g. ``'Zhvi_AllHomes'``.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame indexed by ``(FIPS, date)``.
    """
    df_wide = load_csv(data_dir=data_dir, dataset=dataset, geo="County")
    #    cw = pd.read_csv(data_dir+'CountyCrossWalk_Zillow2.csv')
    cw = load_crosswalk(data_dir=data_dir)

    df_long = pd.melt(
        df_wide, id_vars=df_wide.columns[:7], value_vars=df_wide.columns[7:]
    ).rename(
        columns={
            "variable": "date",
            "value": dataset,
        }
    )

    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long_cw = pd.merge(
        df_long, cw, left_on="RegionID", right_on="CountyRegionID_Zillow"
    )

    return df_long_cw.set_index(["FIPS", "date"])


def load_state(data_dir=default_dir, dataset="Zhvi_AllHomes"):
    """Load Zillow state home value data and merge with state codes.

    Reads the state-level wide CSV, melts it to long format, converts the
    date column, and merges standard state abbreviation codes on the region
    name.

    Parameters
    ----------
    data_dir : str, optional
        Path to the root Zillow data directory.  Defaults to the
        package-configured base directory.
    dataset : str, optional
        Zillow dataset name (file stem), e.g. ``'Zhvi_AllHomes'``.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with state abbreviation and date columns.
    """
    df_wide = load_csv(data_dir=data_dir, dataset=dataset, geo="State")
    df_long = pd.melt(df_wide, id_vars=["RegionName"], value_vars=df_wide.columns[3:])

    state_codes = misc.load("state_codes")
    state_codes["state_name"] = state_codes["state_name"].str.title()

    df_long = pd.merge(
        df_long, state_codes, left_on="RegionName", right_on="state_name"
    )
    df_long = df_long.drop(columns=["RegionName"])
    df_long = df_long.rename(columns={"variable": "date", "value": dataset})
    df_long["date"] = pd.to_datetime(df_long["date"])

    return df_long


# def load_zip(data_dir=default_dir, dataset='Zhvi_AllHomes'):
#
#    df_wide = load_csv(data_dir=data_dir, dataset=dataset, geo='Zip')
#    df_long = pd.melt(df_wide, id_vars=)


def load_crosswalk(data_dir=default_dir):
    """Load the Zillow county-FIPS crosswalk CSV.

    Parameters
    ----------
    data_dir : str, optional
        Path to the root Zillow data directory.  Defaults to the
        package-configured base directory.

    Returns
    -------
    pandas.DataFrame
        Crosswalk table mapping Zillow ``CountyRegionID_Zillow`` to FIPS
        codes and related county identifiers.
    """
    return pd.read_csv(data_dir + "CountyCrossWalk_Zillow2.csv")


def load_csv(data_dir=default_dir, dataset="Zhvi_AllHomes", geo="State"):
    """Read the raw Zillow CSV file for a given geography.

    Constructs the file path as ``<data_dir>/<geo>/<geo>_<dataset>.csv``
    and reads it with Latin-1 encoding.

    Parameters
    ----------
    data_dir : str, optional
        Path to the root Zillow data directory.  Defaults to the
        package-configured base directory.
    dataset : str, optional
        Zillow dataset name (file stem), e.g. ``'Zhvi_AllHomes'``.
    geo : str, optional
        Geography level subdirectory name, e.g. ``'State'``, ``'County'``,
        ``'Zip'``, or ``'Metro'``.

    Returns
    -------
    pandas.DataFrame
        Raw wide-format DataFrame as read from the CSV file.
    """
    df = pd.read_csv(
        data_dir + geo + "/" + geo + "_" + dataset + ".csv", encoding="latin1"
    )

    return df
