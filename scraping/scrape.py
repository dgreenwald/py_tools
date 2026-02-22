from bs4 import BeautifulSoup
import requests
import time

def get_soup(url, delay=1e-4):
    """Fetch and parse the HTML at *url* as a BeautifulSoup object.

    Parameters
    ----------
    url : str
        The URL to fetch.
    delay : float, optional
        Seconds to sleep after the request (politeness throttle).
        Defaults to 1e-4.

    Returns
    -------
    BeautifulSoup or None
        Parsed document, or ``None`` when the server returns a non-200 status.
    """
    req = requests.get(url)
    time.sleep(delay)
    if req.status_code == 200:
        html = req.content
        return BeautifulSoup(html, 'lxml')
    else:
        print("{:d} Error".format(req.status_code))
        return None

def strip_html(url):
    """Strip HTML tags from a URL and return the plain text."""

    soup = get_soup(url)
    if soup is None:
        return soup

    return soup.get_text(strip=True)

# Backward-compatible alias.
stripHTML = strip_html

def url_to_nltk(url, lower=False):
    """Load URL directly into NLTK"""
    try:
        import nltk
    except ImportError as exc:
        raise ImportError("url_to_nltk requires the optional dependency 'nltk'.") from exc

    raw = strip_html(url)
    if raw is None:
        return None

    if lower:
        raw = raw.lower()

    tokens = nltk.word_tokenize(raw)
    return nltk.Text(tokens)
