import warnings

warnings.warn(
    "Importing from py_tools.scrape is deprecated. "
    "Use py_tools.scraping.scrape instead.",
    DeprecationWarning,
    stacklevel=2,
)

from py_tools.scraping.scrape import *  # noqa: F401,F403,E402
