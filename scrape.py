from bs4 import BeautifulSoup
import nltk
import requests

def get_soup(url):
    """Strip HTML tags from a URL"""
    
    req = requests.get(url)
    if req.status_code == 200:
        html = req.content
        return BeautifulSoup(html, 'lxml')
    else:
        return None

def stripHTML(url):
    """Strip HTML tags from a URL"""
    
    soup = get_soup(url)
    if soup is None: return soup
    
    return soup.get_text(strip=True)

def url_to_nltk(url):
    """Load URL directly into NLTK"""

    raw = stripHTML(url)
    if raw is None: return None
    
    tokens = nltk.word_tokenize(raw.lower())
    return nltk.Text(tokens)