import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request as ur
import zipfile

from colloc import in_out
from py_tools.stats import weighted_quantile

save_dir = "/home/dan/data/ahs/"
out_dir = '/home/dan/Dropbox/output/frm/ahs/'

def clean_hist(df, var):

    # Drop bad values
    df_plot = df[[var, 'weight']].dropna()
    for this_var in [var, 'weight']:
        df_plot = df_plot.ix[df_plot[this_var] < np.inf, :]
        df_plot = df_plot.ix[df_plot[this_var] > -np.inf, :]

    return df_plot

def double_hist(df0, df1, label0, label1, var, bins, filetype='png'):

    if var not in df0 or var not in df1:
        return False

    df_plot0 = clean_hist(df0, var)
    df_plot1 = clean_hist(df1, var)

    if len(df_plot0) == 0 or len(df_plot1) == 0:
        return False

    fig = plt.figure()
    plt.hist(df_plot0[var].values, normed=True, bins=bins, alpha=0.5,
             weights=df_plot0['weight'].values, label=str(label0))
    plt.hist(df_plot1[var].values, normed=True, bins=bins, alpha=0.5,
             weights=df_plot1['weight'].values, label=str(label1))
    # plt.xlabel(var)
    plt.legend()

    plt.savefig(out_dir + 'compare_' + var + '_{0}_{1}.{2}'.format(label0, label1, filetype))
    plt.close(fig)

    return True

def make_hist(df, var, bins, ylim=None, filetype='png'):

    if var in df:

        # Drop bad values
        # df_plot = df[[var, 'weight']].dropna()
        # for this_var in [var, 'weight']:
            # df_plot = df_plot.ix[df_plot[this_var] < np.inf, :]
            # df_plot = df_plot.ix[df_plot[this_var] > -np.inf, :]

        df_plot = clean_hist(df, var)

        if len(df_plot) > 0:

            # print(df[var].describe())

            fig = plt.figure()
            plt.hist(df_plot[var].values, normed=True, bins=bins, alpha=0.5,
                     weights=df_plot['weight'].values, label=str(year))
            plt.xlabel(var)
            if ylim is not None:
                plt.ylim((ylim))
            plt.legend()
            plt.savefig(out_dir + var + '_{0}.{1}'.format(year, filetype))
            plt.close(fig)

    return None

def load(year, **kwargs):

    year_str = str(year)
    year_dir = save_dir + year_str + '/'
    short_year = year_str[-2:]

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

    return df

# Download microdata
# if __name__ == "__main__":

    # save_dir = "/home/dan/data/ahs/"

    # for year in range(1973, 2016):

        # year_dir = save_dir + '{}/'.format(year)
        # in_out.makeDir(year_dir)

# # Download file
        # # url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_1973_National_PUF_CSV.zip"
        # url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_{0}_National_PUF_CSV.zip".format(year)
        # filename = year_dir + 'ahs{}.zip'.format(year)
        # ur.urlretrieve(url, filename)

# # Unzip
        # with zipfile.ZipFile(filename, "r") as zip_ref:
            # zip_ref.extractall(year_dir)

if __name__ == "__main__":

    filetype = 'pdf'

    reimport = False

    cap_name_list = ['MORT', 'AMMORT', 'PMT', 'ZINC', 'WEIGHT', 'YRMOR', 'LPRICE', 
                     'HHSAL', 'ZINCN', 'INT', 'INTW']

    cap_name_list += ['AMMRT' + str(val) for val in range(2, 5)]
    cap_name_list += ['PMT' + str(val) for val in range(2, 5)]

    col_names = {
        name : name.lower()
        for name in cap_name_list
    }

    # years = list(range(1973, 1982)) + list(range(1983, 2016, 2))
    years = list(range(1985, 2014, 2))
    # years = list(range(2003, 2014, 2))
    for year in years:

        short_year = int(str(year)[-2:])
        year_str = str(year)
        year_dir = save_dir + year_str + '/'

        if reimport:

            print("\n\nYEAR = {}".format(year))
            df = load(year, 
                      # nrows=1000
                      ).rename(columns=col_names)

            if year <= 1995:
                recent = np.logical_and(df['yrmor'] >= short_year - 2, df['yrmor'] <= short_year)
            else:
                recent = np.logical_and(df['yrmor'] >= year - 2, df['yrmor'] <= year)

            df = df.ix[np.logical_and(recent, df['pmt'] > 0.0), :]
            if 'mort' in df:
                print(df['mort'].value_counts())
                df = df.ix[df['mort'] == 1, :]

        # if recompute:

            # df = df.rename(columns=col_names)

            if 'zinc' in df:
                df.ix[df['zinc'] <= 0.0, 'zinc'] = np.nan
                df['pti'] = 12.0 * df['pmt'] / df['zinc']
                df['lti'] = df['ammort'] / df['zinc']
                df['vti'] = df['lprice'] / df['zinc']

            # if 'zincn' in df:
                # df.ix[df['zincn'] <= 0.0, 'zincn'] = np.nan
                # df['pti_n'] = 12.0 * df['pmt'] / df['zincn']
                # df['lti_n'] = df['ammort'] / df['zincn']

            if 'hhsal' in df:
                df.ix[df['hhsal'] <= 0.0, 'hhsal'] = np.nan
                df['pti_hhsal'] = 12.0 * df['pmt'] / df['zincn']
                df['lti_hhsal'] = df['ammort'] / df['zincn']

            if 'int' in df:
                df['int'] /= 100
            else:
                df = df.rename(columns={'intw' : 'int'})

            print(df['int'].describe())

            # Add variables
            df['ltv'] = df['ammort'] / df['lprice']

            # Print quantiles
            # if 'int' not in df:
                # df = df.rename(columns={'intw' : 'int'})

            df['ammort_total'] = df['ammort']
            df['pmt_total'] = df['pmt']
            for ii in range(2, 5):
                ammrt_var = 'ammrt' + str(ii)
                pmt_var = 'pmt' + str(ii)
                if ammrt_var in df and pmt_var in df:
                    ix = np.logical_and(df[ammrt_var] > 0.0, df[pmt_var] > 0.0)
                    df.ix[ix, 'ammort_total'] += df.ix[ix, ammrt_var]
                    df.ix[ix, 'pmt_total'] += df.ix[ix, pmt_var]

            df['cpti'] = 12.0 * df['pmt_total'] / df['zinc']
            df['clti'] = df['ammort_total'] / df['zinc']
            df['cltv'] = df['ammort_total'] / df['lprice']

            df.to_pickle(year_dir + 'df_recent_{0}.pkl'.format(year))
        # for var in ['int', 'intw']:
            # if var in df:
                # print(df[var].describe())

        else:

            df = pd.read_pickle(year_dir + 'df_recent_{0}.pkl'.format(year))


        with open(out_dir + 'quantiles_{}.txt'.format(year), 'w') as fid:
            quantiles = np.linspace(0.05, 0.95, 19)
            for var in ['pti', 'ltv', 'lti', 'int']:
                wq = weighted_quantile(df[var].values, df['weight'].values, quantiles)
                fid.write('\nWeighted quantiles of {}\n\n'.format(var))
                fid.write(repr(np.vstack((quantiles, wq)).T))
                fid.write('\n')

        if not reimport:

            # old_year = year - 6
            old_year = 1985
            # old_year = 2007

            if old_year >= 1985 and year != old_year:
            # if True:

                old_year_str = str(old_year)
                old_year_dir = save_dir + old_year_str + '/'
                df_old = pd.read_pickle(old_year_dir + 'df_recent_{0}.pkl'.format(old_year))

                # PTI histogram
                bins = np.linspace(0.0, 0.72, 25)
                for var in ['pti', 'cpti']:
                    double_hist(df_old, df, old_year, year, var, bins, filetype=filetype)

                # LTV
                bins = np.linspace(0.5, 1.1, 25)
                for var in ['ltv', 'cltv']:
                    double_hist(df_old, df, old_year, year, var, bins, filetype=filetype)

                # LTI
                bins = np.linspace(0.0, 10.0, 21)
                for var in ['lti', 'clti', 'vti']:
                    double_hist(df_old, df, old_year, year, var, bins, filetype=filetype)

                # Interest rate
                bins = np.arange(21.0)
                for var in ['int']:
                    double_hist(df_old, df, old_year, year, var, bins, filetype=filetype)

        if False:
            # PTI histogram
            bins = np.linspace(0.0, 0.72, 25)
            for var in ['pti', 'pti_hhsal']:
                make_hist(df, var, bins, ylim=(0.0, 6.0), filetype=filetype)

            # LTV
            bins = np.linspace(0.5, 1.1, 25)
            make_hist(df, 'ltv', bins, filetype=filetype)

            # LTI
            bins = np.linspace(0.0, 10.0, 21)
            for var in ['lti', 'lti_hhsal']:
                make_hist(df, var, bins, ylim=(0.0, 0.6), filetype=filetype)

            # Interest rate
            bins = np.arange(21.0)
            for var in ['int', 'intw']:
                make_hist(df, var, bins, filetype=filetype)
