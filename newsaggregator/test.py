# import json  # dumping exporting output
import datetime
import random
from time import struct_time, mktime, strptime, strftime, localtime

# from nltk import SnowballStemmer, WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel

import newsaggregator.news_finder as nf

nf.get_all_news_entries()
news_entry = nf.get_news_from_keywords(['catalan'])[0]
news_entry
related_news = nf.get_related_news(news_entry)
related_news

datetime.datetime.now()

from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
# x = np.random.random(13876)
# x = np.asarray([1,2,3,4,100,101,102,104,200,201,202,4000,4001,4002,4003])



X = []
for (title, description, link, date, named_entities, news_id) in related_news:
    # print(dir(date))
    X.append(mktime(date.timetuple()))

# for i in range(30):
#     X.append(mktime(randomDate("1/1/2008 1:30 AM", "1/3/2008 4:50 PM", random.random())))

X = np.asarray(X)
sorted(X)

max_co = (-1, -1)
for c in range(2, 10):
    km = KMeans(c)
    km.fit(X.reshape(-1,1))  # -1 will be calculated to be 13876 here
    silhouette_coef = metrics.silhouette_score(X.reshape(-1,1), km.labels_, sample_size=1000)
    print((c, silhouette_coef))
    if silhouette_coef > max_co[1]:
        max_co = (c, silhouette_coef)

print(max_co)


km.labels_
# np.shape(km.labels_)
# np.shape(X)
re = []
for (l, x) in zip(km.labels_, X):
    re.append((l, x))

sorted(re)

type(km.labels_)

np.unique(km.labels_)

# struct_time(tm_year=2017, tm_mon=10, tm_mday=25, tm_hour=18, tm_min=39, tm_sec=0, tm_wday=2, tm_yday=298, tm_isdst=0)
date = news_entry[3] # type: struct_time
dt = datetime.datetime.fromtimestamp(mktime(date))
dt
date2 = datetime.datetime.now()
date2-dt
dt

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = mktime(strptime(start, format))
    etime = mktime(strptime(end, format))

    ptime = stime + prop * (etime - stime)

    # return strftime(format, localtime(ptime))
    return localtime(ptime)


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)







# stemmer = SnowballStemmer('english')
# lemmer = WordNetLemmatizer()
# lemmer.lemmatize('presidential')
# lemmer.lemmatize('presidents')
# stemmer.stem('president')
# stemmer.stem('presidental')

# feeds = rf.get_default_feeds()
# news_data = rf.get_feeds_data_2(feeds)
# news_data.values()
#
# lines = []
# for feed, contents in news_data.items():
#     lines.extend(contents)
#
# len(lines)
#
# vectorizer = TfidfVectorizer()
# tf_idf_matrix = vectorizer.fit_transform(lines)
# tf_idf_matrix
#
#
# cosine_similarities = linear_kernel(tf_idf_matrix[0:1], tf_idf_matrix).flatten()
# cosine_similarities
# related_news_indices = cosine_similarities.argsort()
# related_news_indices[-2:-7:-1]
# cosine_similarities[related_news_indices[-2:-7:-1]]
# lines[0]
# lines[79]


# tf_idf_dense_matrix = tf_idf_matrix.todense()
# vectorizer.get_feature_names()

# a = 'Italy investigates anti-Semitic Anne Frank stickers in stadium. Lazio fans are thought to have used the stickers with Anne Frank wearing the jersey of rivals Roma.'
# b = "Jihad: Toulouse boy's name leads to France dilemma. The parents' chosen name is referred to France's state prosecutor for a ruling."
# c = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
# d = 'From Paragon To Pariah: How Kaczynski Is Driving Poland Away from Europe. Jaroslaw Kaczynski, the most powerful politician in Poland, is the architect of judicial reforms that have drawn massive criticism across Europe. As the Polish government chips away at checks and balances, is it possible the politician could drive the country out of the EU?'
# e = "'Nazis, Spies and Terrorists': Can the German-Turkish Relationship Be Saved?. In recent months, relations between Germany and Turkey have reached a new low. After a series of escalating spats, tourism and investment in the country have collapsed. Will it finally drive Turkish President Erdogan to change course?"
# f = 'Rifts EU dispute Deepening Violations rule Poland German Rule Merkel Risks Hungary Chancellor move stop Law Angela exception Brussel agreements practice'

# ne.get_tagged_tree(e)
# ne.get_named_entities_from_text(c)
#
#
# import nltk
# stemmer = nltk.stem.SnowballStemmer('english')
# stemmer.stem('objective')
# [stemmer.stem(w) for w in f.split()]


# with open('phuongs_prototype/feeds', 'w') as outfile:
#     dump = json.dump(news_data, outfile)
