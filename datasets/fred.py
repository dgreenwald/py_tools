import datetime
import pandas as pd
import pandas_datareader.data as web

def load(codes=None, code_names={}, 
         start=datetime.datetime(1900, 1, 1),
         end = datetime.datetime.today()):
    """Load data from FRED, will replace codes with names if code_names is passed as a dict"""

    if codes is None:
        codes = list(code_names.keys())

    # start = datetime.datetime(1900, 1, 1)
    # end = datetime.datetime.today()
    df = web.DataReader(codes, "fred", start, end)
    df.rename(columns=code_names, inplace=True)

    return df
