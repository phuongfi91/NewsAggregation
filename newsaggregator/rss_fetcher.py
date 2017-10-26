from time import mktime
import datetime
import feedparser as fp  # parse feed data
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer  # stemming text data
from nltk.tokenize import word_tokenize  # tokenize text data
from nltk.corpus import stopwords  # removing stop words
from string import punctuation  # get punctuation letters
from html import unescape  # un-escape html text input
from bs4 import BeautifulSoup  # remove html tags
from unidecode import unidecode  # replace fancy unicode characters
from newsaggregator.named_entities import get_named_entities_from_text  # extract named entities


def get_feeds(sources: list):
    return [fp.parse(s) for s in sources]


def get_default_feeds():
    sources = [
        # r'phuongs_prototype/bbc_rss.xml',
        # r'phuongs_prototype/cnn_rss.xml',
        'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
        'http://rss.cnn.com/rss/edition_europe.rss',
        'https://www.cnbc.com/id/19794221/device/rss/rss.html',
        'http://rss.nytimes.com/services/xml/rss/nyt/Europe.xml',
        'http://www.economist.com/sections/europe/rss.xml',
        'http://www.spiegel.de/international/europe/index.rss',

        # erogenous sources:
        # 'http://www.voxeurop.eu/en/topic/europe/feed',
        # 'http://www.eurotopics.net/export/en/rss.xml',
        # 'https://xml.euobserver.com/rss.xml',
    ]
    return get_feeds(sources)


def streamline(line: str, stemmer, lemmer, stop_words):
    processed = clean_text(line)

    named_entities = get_named_entities_from_text(processed)
    named_entities = process_keywords(named_entities, lemmer, stop_words)

    # TODO: fix hardcoded language preference
    tokens = word_tokenize(processed, 'english')
    processed = ' '.join(stemmer.stem(token) for token in tokens if
                         not any((c in punctuation) for c in token) and token not in stop_words)
    return named_entities, processed


def clean_text(line):
    html = BeautifulSoup(line, 'html5lib')
    processed = html.get_text()
    processed = unescape(processed)
    processed = unidecode(processed)
    return processed


def process_keywords(keywords, lemmer=None, stop_words=None):
    if lemmer == None:
        lemmer = WordNetLemmatizer()
    if stop_words == None:
        # TODO: hardcoded language
        stop_words = stopwords.words('english')

    keywords = [lemmer.lemmatize(k).lower() for k in keywords]
    # keywords = [''.join(ch for ch in k if ch not in punctuation) for k in keywords if k not in stop_words]

    return keywords


def get_feeds_data(feeds: list):
    f_data = {}

    # TODO: fix hardcoded language preference
    stemmer = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    lemmer = WordNetLemmatizer()

    for f in feeds:  # type: fp.FeedParserDict
        feed_title = f.feed.title  # type: str
        feed_title = feed_title.lower()
        feed_title = feed_title.replace(' ', '_')
        entries = [
            streamline(e.title + '. ' + e.description if e.get('description') else "", stemmer, lemmer, stop_words)
            for e in f.entries]
        f_data[feed_title] = entries

    return f_data


def get_feeds_data_2(feeds: list):
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
