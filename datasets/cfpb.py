import pandas as pd
from py_tools import time_series as ts

import os
default_dir = os.environ['PY_TOOLS_DATA_DIR'] + 'cfpb/'

def process_data(raw_dir='/data/cfpb', data_dir=default_dir):

    df = pd.read_csv(raw_dir + 'mortgage_complaints.csv')

    df = df.drop(columns=['Product', 'Consumer complaint narrative', 'Complaint ID'])

    for var in ['Sub-product', 'Issue', 'Sub-issue', 'Company public response',
                'Company', 'State', 'Tags', 'Consumer consent provided?',
                'Submitted via', 'Company response to consumer', 
                'Timely response?', 'Consumer disputed?']:

        print("categorizing " + var)
        df[var] = df[var].astype('category')

    for var in ['Date received', 'Date sent to company']:

        df[var] = pd.to_datetime(df[var], format='%m/%d/%y')

    df.to_pickle(data_dir + 'mortgage_complaints.pkl')

    return None

def load(reimport=False, data_dir=default_dir):

    pkl_file = data_dir + 'mortgage_complaints.pkl'

    if reimport or not os.path.exists(pkl_file):
        process_data()

    return pd.read_pickle(pkl_file)


