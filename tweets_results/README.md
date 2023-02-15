**topics.txt**: All 50 topics with top words, each topic shown at -1.0, -0.5, 0.0, 0.5, 1.0 ideal point values since topic has different top words at different political polarization levels.

**interactive_htmls/**: Interactive HTML file to visualize every topic produced by the TBIP model (view any file in a browser - download the .html file first).

**texts_topic_proportions.csv**: Raw texts, their author (Bioguide ID), the timestamp of their tweet, and the topic proportion for each of the topic in that particular tweet (the topic proportions for each tweet sum to 1.0).

**topic_proportions_per_author.csv**: The average proportion of each topic that an author (Bioguide ID) used in their tweets during congresses 115 and 116. The topic proportions for every author sum to 1. This can give an idea as to which topics an author focused more on in their Twitter tweets tweeted during the two congresses.

**bid_to_tbip_tweets.pkl**: Mapping (dict) from legislator BioGuide ID to their ideal point value. 