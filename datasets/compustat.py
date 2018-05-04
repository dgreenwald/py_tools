import pandas as pd

from . import defaults
default_dir = defaults.base_dir() + 'compustat/'

def load(dataset='quarterly', data_dir=default_dir):

    
