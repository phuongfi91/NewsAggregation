# import json  # dumping exporting output
from nltk import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import newsaggregator.news_finder as nf

nf.get_all_news_entries()
news_entry = nf.get_news_from_keywords(['Spain'])[0]
news_entry
nf.get_related_news(news_entry)

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
