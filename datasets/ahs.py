import os
import pandas as pd
# import urllib.request as ur
# import zipfile

# from colloc import in_out

data_dir = "/home/dan/data/ahs/"

def load(year, reimport=False, **kwargs):

    year_str = str(year)
    year_dir = data_dir + year_str + '/'
    short_year = year_str[-2:]

    pkl_file = data_dir + 'ahs_{}.pkl'.format(year)
    if not os.path.exists(pkl_file) or reimport:

        if year <= 1995:
            filename = year_dir + 'tahs{0}n.csv'.format(short_year)
            try:
                df = pd.read_csv(filename, **kwargs) 
            except FileNotFoundError:
                filename = filename.replace('csv', 'CSV')
                df = pd.read_csv(filename, **kwargs) 
        elif year <= 2016:

            if year <= 1999:
                house_file = year_dir + 'thoushld.CSV'
            elif year == 2001:
                house_file = year_dir + 'tnewhouse.CSV'
            elif year <= 2011:
                house_file = year_dir + 'tnewhouse.csv'
            elif year <= 2015:
                house_file = year_dir + 'newhouse.csv'
            else:
                raise Exception

            if year <= 2011:
                mortg_file = year_dir + 'tmortg.csv'
            elif year <= 2015:
                mortg_file = year_dir + 'mortg.csv'

            df = pd.merge(
                pd.read_csv(house_file, **kwargs).rename(columns={'CONTROL' : 'control'}),
                pd.read_csv(mortg_file, **kwargs).rename(columns={'CONTROL' : 'control'}),
                left_on='control',
                right_on='control',
            )

            if year < 2001:
                for extra_file in ['weight', 'toppuf']:
                    extra_filename = year_dir + 't{}.csv'.format(extra_file)
                    df = pd.merge(
                        df, 
                        pd.read_csv(extra_filename).rename(columns={'CONTROL' : 'control'}),
                        left_on='control',
                        right_on='control',
                    )

        df.to_pickle(pkl_file)

    else:

        df = pd.read_pickle(pkl_file)

    return df

# Download microdata
# if __name__ == "__main__":

    # data_dir = "/home/dan/data/ahs/"

    # for year in range(1973, 2016):

        # year_dir = data_dir + '{}/'.format(year)
        # in_out.makeDir(year_dir)

# # Download file
        # # url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_1973_National_PUF_CSV.zip"
        # url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_{0}_National_PUF_CSV.zip".format(year)
        # filename = year_dir + 'ahs{}.zip'.format(year)
        # ur.urlretrieve(url, filename)

# # Unzip
        # with zipfile.ZipFile(filename, "r") as zip_ref:
            # zip_ref.extractall(year_dir)

