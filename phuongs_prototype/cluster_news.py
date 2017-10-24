import re
import string
import feedparser as fp
import numpy as np
import pandas as pd
import sklearn.cluster as cl
import distance
from nltk.stem.snowball import SnowballStemmer
import html
import json


def get_feeds(sources: list):
    return [fp.parse(s) for s in sources]


def streamline(line: str, regex, stemmer, stop_words):
    line = html.unescape(line)
    line = regex.sub('', line)
    line = ''.join(ch for ch in line if ch not in set(string.punctuation))
    line = ' '.join(stemmer.stem(word) for word in line.split() if word not in stop_words)
    return line


def get_feeds_data(feeds: list):
    f_data = {}

    stemmer = SnowballStemmer('english')
    tag_remover = re.compile('<[^>]*>')
    stop_words = pd.read_csv('stop-word-list.csv')
    stop_words = [x.strip() for x in list(stop_words)]

    for f in feeds:  # type: fp.FeedParserDict
        feed_title = f.feed.title  # type: str
        feed_title = feed_title.lower()
        feed_title = feed_title.replace(' ', '_')
        entries = [
            streamline(e.title + ' ' + e.description if e.get('description') else "", tag_remover, stemmer, stop_words)
            for e in f.entries]
        f_data[feed_title] = entries

    return f_data


news = get_feeds([
    # r'phuongs_prototype/bbc_rss.xml',
    # r'phuongs_prototype/cnn_rss.xml'
    'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
    'http://rss.cnn.com/rss/edition_europe.rss',
    # 'http://www.eurotopics.net/export/en/rss.xml',
    'https://www.cnbc.com/id/19794221/device/rss/rss.html',
    'http://rss.nytimes.com/services/xml/rss/nyt/Europe.xml',
    'http://www.voxeurop.eu/en/topic/europe/feed',
    'https://xml.euobserver.com/rss.xml',
    'http://www.economist.com/sections/europe/rss.xml',
    'http://www.spiegel.de/international/europe/index.rss',
])

news_data = get_feeds_data(news)
news_data

with open('phuongs_prototype/feeds', 'w') as outfile:
    dump = json.dump(news_data, outfile)

# vectorizer = TfidfVectorizer()
# tf_idf_matrix = vectorizer.fit_transform( [ b, c ])
# tf_idf_dense_matrix = tf_idf_matrix.todense()
# vectorizer.get_feature_names()

lines = []
for feed, contents in news_data.items():
    lines.extend(contents)

lines = np.asarray(lines)  # So that indexing with a list will work
len(lines)

lev_similarity = -1 * np.array([[distance.levenshtein(l1, l2) for l1 in lines] for l2 in lines])

affprop = cl.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)

kmean = cl.KMeans()
kmean.fit(lev_similarity)

for cluster_id in np.unique(affprop.labels_):
    print(cluster_id)
    # exemplar = lines[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(lines[np.nonzero(affprop.labels_ == cluster_id)])
    cluster_str = " \n".join(cluster)
    # print(" - *%s:* %s" % (exemplar, cluster_str))
    print(cluster_str)

for cluster_id in np.unique(kmean.labels_):
    print(cluster_id)
    # exemplar = lines[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(lines[np.nonzero(kmean.labels_ == cluster_id)])
    cluster_str = " \n".join(cluster)
    # print(" - *%s:* %s" % (exemplar, cluster_str))
    print(cluster_str)


#
# categories = [
#     'alt.atheism',
#     'talk.religion.misc',
#     'comp.graphics',
#     'sci.space',
# ]
# import numpy as np
# from sklearn.datasets import fetch_20newsgroups
# dataset = fetch_20newsgroups(subset='all', categories=categories,
#                              shuffle=True, random_state=42)
# # np.unique(dataset.target)
# dataset.data
#
# import json
# with open('phuongs_prototype/feeds', 'w') as outfile:
#     dump = json.dump(dataset.data, outfile)
# type(dataset)
