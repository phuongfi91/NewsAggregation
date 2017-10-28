from time import mktime  # parse news published date
import datetime  # parse news published date
import feedparser as fp  # parse feed data
from nltk import WordNetLemmatizer  # lemmatize keywords
from nltk.stem.snowball import SnowballStemmer  # stemming text data
from nltk.tokenize import word_tokenize  # tokenize text data
from nltk.corpus import stopwords  # removing stop words
from string import punctuation  # get punctuation letters
from html import unescape  # un-escape html text input
from bs4 import BeautifulSoup  # remove html tags
from unidecode import unidecode  # replace fancy unicode characters
from newsaggregator.named_entities import get_named_entities_from_text  # extract named entities


def get_feeds(sources: list):
    """
    Get RSS feed objects from a list of RSS URLs
    :param sources:
    :return:
    """
    return [fp.parse(s) for s in sources]


def get_default_feeds():
    """
    Get RSS feed objects from the default manually collected list of RSS URLs
    :return:
    """
    sources = [
        'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
        'http://rss.cnn.com/rss/edition_europe.rss',
        'https://www.cnbc.com/id/19794221/device/rss/rss.html',
        'http://rss.cnn.com/rss/edition_asia.rss',
        'http://rss.cnn.com/rss/edition_americas.rss',
        'http://rss.cnn.com/rss/edition_africa.rss',
        'http://rss.cnn.com/rss/edition_meast.rss',
        'http://rss.cnn.com/rss/edition_us.rss',
        'http://feeds.bbci.co.uk/news/world/africa/rss.xml',
        'http://feeds.bbci.co.uk/news/world/asia/rss.xml',
        'http://feeds.bbci.co.uk/news/world/latin_america/rss.xml',
        'http://feeds.bbci.co.uk/news/world/middle_east/rss.xml',
        'http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml',
        'https://www.cnbc.com/id/19832390/device/rss/rss.html',
        'https://www.cnbc.com/id/15837362/device/rss/rss.html',
        'https://www.cnbc.com/id/100727362/device/rss/rss.html',
        'http://www.economist.com/feeds/print-sections/73/asia.xml',
        'http://www.economist.com/feeds/print-sections/99/middle-east-africa.xml',
        'http://www.economist.com/feeds/print-sections/72/the-americas.xml',
        'http://www.economist.com/feeds/print-sections/75/europe.xml',
        'http://www.economist.com/feeds/print-sections/71/united-states.xml',

        # 'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
        # 'http://rss.cnn.com/rss/edition_europe.rss',
        # 'https://www.cnbc.com/id/19794221/device/rss/rss.html',
        # 'http://rss.nytimes.com/services/xml/rss/nyt/Europe.xml',
        # 'http://www.economist.com/sections/europe/rss.xml',
        # 'http://www.spiegel.de/international/europe/index.rss',

        # erogenous sources:
        # 'http://www.voxeurop.eu/en/topic/europe/feed',
        # 'http://www.eurotopics.net/export/en/rss.xml',
        # 'https://xml.euobserver.com/rss.xml',
    ]
    return get_feeds(sources)


def streamline(line: str, stemmer, lemmer, stop_words):
    """
    Streamline or process a string in order to get named entities and processable text
    :param line:
    :param stemmer:
    :param lemmer:
    :param stop_words:
    :return:
    """
    processed = clean_text(line)

    named_entities = get_named_entities_from_text(processed)
    named_entities = process_keywords(named_entities, lemmer, stop_words)

    # TODO: fix hardcoded language preference
    tokens = word_tokenize(processed, 'english')
    processed = ' '.join(stemmer.stem(token) for token in tokens if
                         not any((c in punctuation) for c in token) and token not in stop_words)
    return named_entities, processed


def clean_text(line):
    """
    Clean a string extracted from HTML source
    :param line:
    :return:
    """
    html = BeautifulSoup(line, 'html5lib')
    processed = html.get_text()
    processed = unescape(processed)
    processed = unidecode(processed)
    return processed


def process_keywords(keywords, lemmer=None, stop_words=None):
    """
    Lemmatize keywords and potentially remove stop words
    :param keywords:
    :param lemmer:
    :param stop_words:
    :return:
    """
    if lemmer == None:
        lemmer = WordNetLemmatizer()
    if stop_words == None:
        # TODO: hardcoded language
        stop_words = stopwords.words('english')

    keywords = [lemmer.lemmatize(k).lower() for k in keywords]
    # keywords = [''.join(ch for ch in k if ch not in punctuation) for k in keywords if k not in stop_words]

    return keywords


def get_feeds_data(feeds: list):
    """
    Return news data for selected feeds
    :param feeds:
    :return:
    """
    f_data = {}

    # TODO: fix hardcoded language preference
    stemmer = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    lemmer = WordNetLemmatizer()

    for f in feeds:  # type: fp.FeedParserDict
        for e in f.entries:
            title = clean_text(e.title) if e.get('title') else ''
            description = clean_text(e.description) if e.get('description') else ''
            link = e.link if e.get('link') else 'https://google.com'
            date = datetime.datetime.fromtimestamp(mktime(e.published_parsed)) if e.get('published_parsed') else None
            named_entities, processed = streamline(title + '. ' + description, stemmer, lemmer, stop_words)
            entry_id = hash(processed)
            f_data[entry_id] = (title, description, link, date, named_entities, processed)

    return f_data
