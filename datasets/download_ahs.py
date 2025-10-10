import urllib.request
import urllib.error
import zipfile

from colloc import in_out

def download_if_exists(url, filename):

    # status = urllib.request.urlopen(url).getcode()
    # if status == 200:
    try:
        urllib.request.urlretrieve(url, filename)
        return True
    except urllib.error.HTTPError as e:
        print(e)
        return False

# save_dir = "/home/dan/data/ahs/"
save_dir = '/data/ahs/'

# Note: no national file in 1982
years = list(range(1973, 1982)) + list(range(1983, 2016, 2))

for year in years:

    year_str = str(year)
    year_dir = save_dir + year_str + '/'
    short_year = year_str[-2:]

    print('\n\nYEAR = {}'.format(year))

# Download file
    if year <= 1983:
        url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_{0}_National_PUF_CSV.zip".format(year)
    elif year <= 2009:
        url = "http://www2.census.gov/programs-surveys/ahs/{0}/AHS%20{0}%20National%20PUF%20v1.1%20CSV.zip".format(year)
    elif year == 2011:
        url = "http://www2.census.gov/programs-surveys/ahs/2011/AHS%202011%20National%20and%20Metropolitan%20PUF%20v1.4%20CSV.zip"
    elif year == 2013:
        url = "http://www2.census.gov/programs-surveys/ahs/2013/AHS%202013%20National%20PUF%20v1.2%20CSV.zip"
    elif year == 2015:
        url = "http://www2.census.gov/programs-surveys/ahs/2015/AHS%202015%20National%20PUF%20v1.0%20CSV.zip"
    else:
        raise Exception

    in_out.makeDir(year_dir)
    filename = year_dir + 'ahs{}.zip'.format(year)

    # Deal with case of V
    downloaded = download_if_exists(url, filename)
    print("first attempt: downloaded = {}".format(downloaded))
    print("url = {}".format(url))
    if not downloaded:
        url = url.replace("v1", "V1")
        downloaded = download_if_exists(url, filename)
        print("second attempt: downloaded = {}".format(downloaded))
        print("url = {}".format(url))
    if not downloaded:
        # download failed
        raise Exception

# Unzip
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(year_dir)
