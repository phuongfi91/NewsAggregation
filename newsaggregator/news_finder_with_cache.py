from time import mktime  # turn time into 1-dimensional numerical data

import datetime
from sklearn.feature_extraction.text import TfidfVectorizer  # TF/IDF
from sklearn.metrics.pairwise import linear_kernel  # for cosine similarity calc
from sklearn.cluster import KMeans  # K-Means clustering method
from sklearn import metrics  # Silhouette coefficient for measuring clustering performance
import newsaggregator.rss_fetcher as rf
import numpy as np


class NewsFinder:
    def __init__(self):
        self.last_cache_timestamp = datetime.datetime.min

    def get_news_from_keywords(self, keywords):
        """
        Extract news from user input keywords
        :param keywords:
        :return:
        """
        keywords = rf.process_keywords(keywords)

        news_data = self.get_all_news_entries()

        news_entries = []
        for title, description, link, date, named_entities, processed in news_data.values():
            if all((k in named_entities) for k in keywords):
                news_entries.append((title, description, link, date, named_entities, hash(processed)))

        return news_entries

    def get_related_news(self, news_entry, top_entries=10):
        """
        Get news related to a specific news article
        :param news_entry:
        :param top_entries:
        :return:
        """
        target_id = news_entry[-1]
        news_data = self.get_all_news_entries()
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


        if not self.tf_idf_matrix:
            vectorizer = TfidfVectorizer()
            self.tf_idf_matrix = vectorizer.fit_transform(processed_text)

        cosine_similarities = linear_kernel(self.tf_idf_matrix[target_index:target_index + 1], self.tf_idf_matrix).flatten()
        related_news_indices = cosine_similarities.argsort()[-2:-2 - top_entries:-1]

        news_entries_hashes = []
        for i in related_news_indices:
            news_entries_hashes.append(news_entries_mapping[i])

        news_entries = []
        for h in news_entries_hashes:
            title, description, link, date, named_entities, processed = news_data[h]
            news_entries.append((title, description, link, date, named_entities, hash(processed)))

        return news_entries

    def get_all_news_entries(self):
        """
        Self-explanatory
        :return:
        """
        if datetime.datetime.now() - self.last_cache_timestamp > datetime.timedelta(minutes=30):
            self.last_cache_timestamp = datetime.datetime.now()
            self.feeds = rf.get_default_feeds()
            self.feeds_data = rf.get_feeds_data(self.feeds)
            self.tf_idf_matrix = None

        return self.feeds_data

    def get_news_categorical_labels(self, news_entries):
        """
        Categorize the selected list of news into separate groups
        :param news_entries:
        :return:
        """
        X = []
        for (title, description, link, date, named_entities, news_id) in news_entries:
            X.append(mktime(date.timetuple()))

        X = np.asarray(X).reshape(-1, 1)

        max_silhouette_coef = (-1, -1, None)
        for n_clusters in range(2, len(news_entries)):
            km = KMeans(n_clusters)
            km.fit(X)
            silhouette_coef = metrics.silhouette_score(X, km.labels_, sample_size=1000)
            if silhouette_coef > max_silhouette_coef[1]:
                max_silhouette_coef = (n_clusters, silhouette_coef, km.labels_)

        return list(max_silhouette_coef[-1])
