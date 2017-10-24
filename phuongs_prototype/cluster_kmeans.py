import re
import string
import feedparser as fp
import numpy as np
import pandas as pd
import sklearn.cluster as cl
import distance
import unicodedata
from nltk.stem.snowball import SnowballStemmer
import html
import json
import phuongs_prototype.named_entities_extractor as ne


def get_feeds(sources: list):
    return [fp.parse(s) for s in sources]


def streamline(line: str, regex, stemmer, stop_words):
    line = html.unescape(line)
    line = regex.sub(' ', line)
    # line = ''.join(ch for ch in line if ch not in set(string.punctuation))
    # line = ' '.join(stemmer.stem(word) for word in line.split() if word not in stop_words)

    line = unicodedata.normalize("NFKD", line)
    # line = ne.get_continuous_chunks(line)
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
    # 'https://www.cnbc.com/id/19794221/device/rss/rss.html',
    # 'http://rss.nytimes.com/services/xml/rss/nyt/Europe.xml',
    # 'http://www.voxeurop.eu/en/topic/europe/feed',
    # 'http://www.economist.com/sections/europe/rss.xml',
    # 'http://www.spiegel.de/international/europe/index.rss',

    # 'http://www.eurotopics.net/export/en/rss.xml',
    # 'https://xml.euobserver.com/rss.xml',
])

news_data = get_feeds_data(news)
news_data
type(news_data)

with open('phuongs_prototype/feeds', 'w') as outfile:
    dump = json.dump(news_data, outfile)

# vectorizer = TfidfVectorizer()
# tf_idf_matrix = vectorizer.fit_transform( [ b, c ])
# tf_idf_dense_matrix = tf_idf_matrix.todense()
# vectorizer.get_feature_names()

lines = []
for feed, contents in news_data.items():
    lines.extend(contents)

lines

lines = np.asarray(lines)  # So that indexing with a list will work
lines_list = lines.tolist()
type(lines)
lines_list


# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)


print("%d documents" % len(dataset.data))
# print("%d documents" % len(lines_list))
# print("%d categories" % len(dataset.target_names))
print()

# labels = dataset.target
# true_k = np.unique(labels).shape[0]
# true_k = 10

max_silhouette_coef = (0, -1)
for true_k in range(2, 10):

    # true_k = 7
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)
    # X = vectorizer.fit_transform(lines_list)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    # #############################################################################
    # Do the actual clustering

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(labels, km.labels_))
    silhouette_coef = metrics.silhouette_score(X, km.labels_, sample_size=100000)

    if silhouette_coef > max_silhouette_coef[1]:
        max_silhouette_coef = (true_k, silhouette_coef)

    print("Silhouette Coefficient: %0.3f"
          % silhouette_coef)

    print()


    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()


print("Max Silhouette true_k: %d" % max_silhouette_coef[0])
print("Max Silhouette Coefficient: %0.3f" % max_silhouette_coef[1])
