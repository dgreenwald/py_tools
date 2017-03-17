import pandas as pd

# def load():
if __name__ == "__main__":

    data_dir = '/data/ahs/2013/'
    # infile = 'mortg.csv' 
    infile = 'newhouse.csv'

    df = pd.read_table(data_dir + infile, sep=',')
