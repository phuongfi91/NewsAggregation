import newsaggregator.news_finder as nf

nf.get_all_news_entries()
len(nf.get_all_news_entries())
news_entry = nf.get_news_from_keywords(['london'])[0]
news_entries = nf.get_news_from_keywords(['brexit'])
news_entries = nf.get_news_from_keywords(['catalonia'])

nf.get_news_from_keywords(['trump'])

news_entry

for title, description, link, date, named_entities, news_id in news_entries:
    print(title + ". " + description)
    print()

related_news = nf.get_related_news(news_entry)
related_news
nf.get_news_categorical_labels(related_news)

# ==========================================================================

import newsaggregator.news_finder_with_cache as nf

news_finder = nf.NewsFinder()

news_finder.get_all_news_entries()

len(news_finder.get_all_news_entries())
news_entry = news_finder.get_news_from_keywords(['london'])[0]
news_entry
news_entries = news_finder.get_news_from_keywords(['brexit'])
news_entries = news_finder.get_news_from_keywords(['catalonia'])
news_entries

news_finder.get_news_from_keywords(['trump'])

related_news = news_finder.get_related_news(news_entry)
related_news
news_finder.get_news_categorical_labels(related_news)
