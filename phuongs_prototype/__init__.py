import re
import string
import feedparser as fp
import numpy as np
import pandas as pd
import sklearn.cluster as cl
import distance
from nltk.stem.snowball import SnowballStemmer


def get_feeds(sources: list):
    return [fp.parse(s) for s in sources]


def streamline(line: str, regex, stemmer, stop_words):
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
        entries = [streamline(e.title + ' ' + e.description, tag_remover, stemmer, stop_words) for e in f.entries]
        f_data[feed_title] = entries

    return f_data


news = get_feeds([
    r'phuongs_prototype/bbc_rss.xml',
    r'phuongs_prototype/cnn_rss.xml'
])

news_data = get_feeds_data(news)

# vectorizer = TfidfVectorizer()
# tf_idf_matrix = vectorizer.fit_transform( [ b, c ])
# tf_idf_dense_matrix = tf_idf_matrix.todense()
# vectorizer.get_feature_names()

lines = []
for feed, contents in news_data.items():
    lines.extend(contents)

lines = np.asarray(lines)  # So that indexing with a list will work

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
