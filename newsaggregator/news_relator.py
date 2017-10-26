# import json  # dumping exporting output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import newsaggregator.rss_fetcher as rf
import newsaggregator.named_entities as ne

feeds = rf.get_default_feeds()
news_data = rf.get_feeds_data_2(feeds)
news_data

lines = []
for feed, contents in news_data.items():
    lines.extend(contents)

len(lines)

vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(lines)
tf_idf_matrix


cosine_similarities = linear_kernel(tf_idf_matrix[0:1], tf_idf_matrix).flatten()
cosine_similarities
related_news_indices = cosine_similarities.argsort()
related_news_indices[-2:-7:-1]
cosine_similarities[related_news_indices[-2:-7:-1]]
lines[0]
lines[79]
