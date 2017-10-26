from time import mktime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import newsaggregator.rss_fetcher as rf


def get_news_from_keywords(keywords):
    keywords = rf.process_keywords(keywords)

    news_data = get_all_news_entries()

    news_entries = []
    for title, description, link, date, named_entities, processed in news_data.values():
        if all((k in named_entities) for k in keywords):
            news_entries.append((title, description, link, date, named_entities, hash(processed)))

    return news_entries


def get_related_news(news_entry, top_entries=10):
    target_id = news_entry[-1]
    news_data = get_all_news_entries()
    target_index = -1

    processed_text = []
    news_entries_mapping = {}
    for id, news in news_data.items():
        item_index = len(news_entries_mapping)
        news_entries_mapping[item_index] = id
        if id == target_id:
            target_index = item_index
        processed_text.append(news[-1])

    if target_index == -1:
        return None

    # TODO: cache this matrix
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(processed_text)

    cosine_similarities = linear_kernel(tf_idf_matrix[target_index:target_index + 1], tf_idf_matrix).flatten()
    related_news_indices = cosine_similarities.argsort()[-2:-2 - top_entries:-1]
    # print(cosine_similarities[related_news_indices])

    news_entries_hashes = []
    for i in related_news_indices:
        news_entries_hashes.append(news_entries_mapping[i])

    news_entries = []
    for h in news_entries_hashes:
        title, description, link, date, named_entities, processed = news_data[h]
        news_entries.append((title, description, link, date, named_entities, hash(processed)))

    return news_entries


def get_all_news_entries():
    feeds = rf.get_default_feeds()
    return rf.get_feeds_data_2(feeds)


def get_news_categorical_labels(news_entries):
    X = []
    for (title, description, link, date, named_entities, news_id) in news_entries:
        X.append(mktime(date.timetuple()))

    X = np.asarray(X).reshape(-1, 1)

    max_silhouette_coef = (-1, -1, None)
    for n_clusters in range(2, 10):
        km = KMeans(n_clusters)
        km.fit(X)
        silhouette_coef = metrics.silhouette_score(X, km.labels_, sample_size=1000)
        # print((n_clusters, silhouette_coef))
        if silhouette_coef > max_silhouette_coef[1]:
            max_silhouette_coef = (n_clusters, silhouette_coef, km.labels_)

    # print(max_silhouette_coef)

    return list(max_silhouette_coef[-1])
