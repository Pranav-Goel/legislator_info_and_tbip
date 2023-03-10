"""Convert Senate tweets from 2020 Democratic candidates to bag of words.

This requires that a file containing the tweets called `tweets.csv` be stored
in  `data/candidate-tweets-2020/raw/`.
"""

import os

import numpy as np
import pandas as pd
from scipy import sparse
import setup_utils as utils
from sklearn.feature_extraction.text import CountVectorizer

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
data_dir = os.path.join(project_dir, "data/tweets_cong_115_116_sampled/")
save_dir = os.path.join(project_dir, "data/tweets_cong_115_116_sampled/clean")

df = pd.read_csv(os.path.join(data_dir, "sampled_twitter_data.csv")) #"all_tweets_df.csv"))

candidates = list(df['Author ID'])
tweets = list(df['Text'])

candidate_to_candidate_id = dict(
    [(y, x) for x, y in enumerate(sorted(set(candidates)))])
author_indices = np.array(
    [candidate_to_candidate_id[s] for s in candidates])
author_map = np.array(list(candidate_to_candidate_id.keys()))

stopwords = set(np.loadtxt(
    os.path.join(project_dir, "setup/stopwords/candidate_tweets.txt"),
    dtype=str,
    delimiter="\n")).union(set(np.loadtxt(os.path.join(project_dir, "setup/stopwords/mallet_plus_new_legis_proc_jargon_stopwords.txt"), dtype=str, delimiter="\n")))

count_vectorizer = CountVectorizer(#min_df=0.0005, 
                                   max_df=0.95, 
                                   ngram_range=(1, 3),
                                   stop_words=stopwords, 
                                   token_pattern="[a-zA-Z#]+",
                                   max_features=20000)
# Learn initial document term matrix to identify words to exclude based
# on author counts.
counts = count_vectorizer.fit_transform(tweets)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])
print(len(vocabulary))
# Remove phrases spoken by only 1 candidate.
#counts_per_author = utils.bincount_2d(author_indices, counts.toarray())
#min_authors_per_word = 2
#author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
#acceptable_words = np.where(
    #author_counts_per_word >= min_authors_per_word)[0]

# Fit final document-term matrix with new vocabulary.
#count_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                   #vocabulary=vocabulary[acceptable_words],
                                   #token_pattern="[a-zA-Z#]+",
                                   #max_features=20000)
#counts = count_vectorizer.fit_transform(tweets)
#vocabulary = np.array(
#[k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
#                            key=lambda kv: kv[1])])

# Adjust counts by removing unigram/n-gram pairs which co-occur.
#counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)
counts = sparse.csr_matrix(counts)
print(counts.shape)
# Save data.
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# `counts.npz` is a [num_documents, num_words] sparse matrix containing the
# word counts for each document.
sparse.save_npz(os.path.join(save_dir, "counts.npz"),
                sparse.csr_matrix(counts).astype(np.float32))
# `author_indices.npy` is a [num_documents] vector where each entry is an
# integer indicating the author of the corresponding document.
np.save(os.path.join(save_dir, "author_indices.npy"), author_indices)
# `vocabulary.txt` is a [num_words] vector where each entry is a string
# denoting the corresponding word in the vocabulary.
np.savetxt(os.path.join(save_dir, "vocabulary.txt"), vocabulary, fmt="%s")
# `author_map.txt` is a [num_authors] vector of strings providing the name of
# each author in the corpus.
np.savetxt(os.path.join(save_dir, "author_map.txt"), author_map, fmt="%s")
# `raw_documents.txt` contains all the documents we ended up using.
stripped_tweets = [tweet.replace("\n", ' ').replace("\r", ' ') 
                   for tweet in tweets]
f = open(os.path.join(save_dir, "raw_documents.txt"), 'w')
for x in stripped_tweets:
    if x!='' and x!=' ':
        f.write(x)
        f.write('\n')
f.close()
