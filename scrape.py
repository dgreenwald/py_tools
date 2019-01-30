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
    # elif req.status_code == 429:
        # print("429 Error")
        # print("got 429, delaying {:g}s".format(delay))
        # time.sleep(delay)
        # new_delay = 2.0 * delay
        # return get_soup(url, delay=new_delay)
    else:
        print("{:d} Error".format(req.status_code))
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
    
    tokens = nltk.word_tokenize(raw)
    return nltk.Text(tokens)
