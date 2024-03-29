from bs4 import BeautifulSoup
import nltk
import requests
import time

def get_soup(url, delay=1e-4):
    """Strip HTML tags from a URL"""
    
    # req = requests.get(url, headers={'User-agent': 'Super Bot 9000'})
    req = requests.get(url)
    if req.status_code == 200:
        html = req.content
        return BeautifulSoup(html, 'lxml')
    else:
        print("{:d} Error".format(req.status_code))
        return None

def stripHTML(url):
    """Strip HTML tags from a URL"""
    
    soup = get_soup(url)
    if soup is None: return soup
    
    return soup.get_text(strip=True)

def url_to_nltk(url, lower=False):
    """Load URL directly into NLTK"""

    raw = stripHTML(url)
    if raw is None: return None
    
    if lower:
        raw = raw.lower()

    tokens = nltk.word_tokenize(raw)
    return nltk.Text(tokens)
