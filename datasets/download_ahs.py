import urllib.request
import urllib.error
import zipfile

from py_tools.in_out import core as in_out

def download_if_exists(url, filename):
    """Attempt to download a URL to a local file.

    Parameters
    ----------
    url : str
        URL of the resource to download.
    filename : str
        Local file path where the downloaded content will be saved.

    Returns
    -------
    bool
        ``True`` if the download succeeded, ``False`` if an
        :class:`urllib.error.HTTPError` was raised.
    """

    # status = urllib.request.urlopen(url).getcode()
    # if status == 200:
    try:
        urllib.request.urlretrieve(url, filename)
        return True
    except urllib.error.HTTPError as e:
        print(e)
        return False

DEFAULT_SAVE_DIR = '/data/ahs/'
# Note: no national file in 1982
DEFAULT_YEARS = list(range(1973, 1982)) + list(range(1983, 2016, 2))


def _get_ahs_url(year):
    if year <= 1983:
        return "http://www2.census.gov/programs-surveys/ahs/{0}/AHS_{0}_National_PUF_CSV.zip".format(year)
    if year <= 2009:
        return "http://www2.census.gov/programs-surveys/ahs/{0}/AHS%20{0}%20National%20PUF%20v1.1%20CSV.zip".format(year)
    if year == 2011:
        return "http://www2.census.gov/programs-surveys/ahs/2011/AHS%202011%20National%20and%20Metropolitan%20PUF%20v1.4%20CSV.zip"
    if year == 2013:
        return "http://www2.census.gov/programs-surveys/ahs/2013/AHS%202013%20National%20PUF%20v1.2%20CSV.zip"
    if year == 2015:
        return "http://www2.census.gov/programs-surveys/ahs/2015/AHS%202015%20National%20PUF%20v1.0%20CSV.zip"
    raise ValueError("Unsupported AHS year: {}".format(year))


def download_all(save_dir=DEFAULT_SAVE_DIR, years=None):
    if years is None:
        years = DEFAULT_YEARS

    for year in years:
        year_str = str(year)
        year_dir = save_dir + year_str + '/'
        filename = year_dir + 'ahs{}.zip'.format(year)

        print('\n\nYEAR = {}'.format(year))

        url = _get_ahs_url(year)
        in_out.make_dir(year_dir)

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
            raise RuntimeError("Failed to download AHS file for year {}".format(year))

        # Unzip
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(year_dir)
