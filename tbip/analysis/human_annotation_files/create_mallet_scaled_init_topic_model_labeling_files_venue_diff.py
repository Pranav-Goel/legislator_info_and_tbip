#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import scipy.sparse as sparse
from scipy.stats import bernoulli, poisson
import analysis_utils_mine as utils

import json
import pandas as pd
import ast
from datetime import datetime
import torch
import pandas as pd
from datetime import datetime, timedelta
import pickle

import matplotlib.pyplot as plt

import os


'''
below, we load the required data for floor speeches and then for tweets - this code is specific to the initial topic modeling output (mallet+scaling) that we used in the TBIP model, BUT can work with any topic modeling output (with modifications in code), all that is needed is: 

1. vocabulary (list of tokens)
2. a topic_word 2-d numpy array (number of topics X vocab size)
3. a document_topic 2-d numpy array (number of documents X number of topics)
4. raw document texts (a list of strings)

everything below is done for both floor speeches and for tweets. 
'''


# floor speeches

project_dir = os.path.abspath('/workspace/pranav/tbip/data/floor_speeches_congs_115_116/') 
fit_dir = os.path.join(project_dir, "mallet_fits_removed_procedural_speeches")

data_dir = os.path.join(project_dir, "clean_removing_procedural")
(_, vocabulary_speeches, _, 
 _) = utils.load_text_data(data_dir)

#print(len(vocabulary_speeches))

topic_word_speeches = np.load(os.path.join(fit_dir, 
                                           'topic_word.npy'))
#print(topic_word_speeches.shape)
doc_topic_speeches = np.load(os.path.join(fit_dir, 
                                          'doc_topic.npy'))
#print(doc_topic_speeches.shape)
raw_texts_speeches = open(os.path.join(data_dir, 'raw_documents.txt')).readlines()
raw_texts_speeches = list(map(lambda x:x.rstrip(), raw_texts_speeches))
#print(len(raw_texts_speeches))


# twitter

project_dir = os.path.abspath('/workspace/pranav/tbip/data/tweets_cong_115_116/') 
fit_dir = os.path.join(project_dir, "mallet_results/tbip_expanded_preprocessing_k50")

data_dir = os.path.join(project_dir, "clean2")
(_, vocabulary_tweets, _, 
 _) = utils.load_text_data(data_dir)

#print(len(vocabulary_tweets))

topic_word_tweets = np.load(os.path.join(fit_dir, 
                                           'topic_word.npy'))
#print(topic_word_tweets.shape)
doc_topic_tweets = np.load(os.path.join(fit_dir, 
                                          'doc_topic.npy'))
#print(doc_topic_tweets.shape)
raw_texts_tweets = open(os.path.join(data_dir, 'raw_documents.txt')).readlines()
raw_texts_tweets = list(map(lambda x:x.rstrip(), raw_texts_tweets))
#print(len(raw_texts_tweets))


vocabulary_tweets = list(vocabulary_tweets)
vocabulary_speeches = list(vocabulary_speeches)


def rescale_to_probs_renorm(arr):
    return arr/arr.sum(1, keepdims=True)


# rescaling to make them probs 
topic_word_speeches = rescale_to_probs_renorm(topic_word_speeches)
#print(topic_word_speeches.shape)

doc_topic_speeches = rescale_to_probs_renorm(doc_topic_speeches)
#print(doc_topic_speeches.shape)

topic_word_tweets = rescale_to_probs_renorm(topic_word_tweets)
#print(topic_word_tweets.shape)

doc_topic_tweets = rescale_to_probs_renorm(doc_topic_tweets)
#print(doc_topic_tweets.shape)

def create_doc_topic_file_for_annotators(doc_topic,
                                         raw_texts,
                                         outpath,
                                         top_doc_num_per_topic=500):
    out_df = pd.DataFrame()
    num_docs, num_topics = doc_topic.shape
    all_top_doc_inds = set()
    for topic_ind in range(num_topics):
        topic_vals = list(enumerate(list(doc_topic[:, topic_ind])))
        topic_vals = sorted(topic_vals, key=lambda x:x[1])[::-1][:top_doc_num_per_topic]
        for ind, _ in topic_vals:
            all_top_doc_inds.add(ind)
    all_top_doc_inds = list(all_top_doc_inds)
    selected_raw_texts = [x for i,x in enumerate(raw_texts) if i in all_top_doc_inds]
    #print(len(all_top_doc_inds))
    out_df['docID'] = [i + 1 for i in range(len(all_top_doc_inds))]
    for topic_ind in range(num_topics):
        out_df['Topic ' + str(topic_ind + 1)] = list(doc_topic[all_top_doc_inds, topic_ind])
    out_df['text'] = selected_raw_texts
    
    out_df.to_excel(os.path.join(outpath, 'document_topics.xlsx'), index=False, float_format='%.3f')


create_doc_topic_file_for_annotators(doc_topic_speeches, 
                                     raw_texts_speeches,
                                     '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/floor_speeches/mallet_output_labeling/')


create_doc_topic_file_for_annotators(doc_topic_tweets,
                                     raw_texts_tweets,
                                     '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/twitter/mallet_output_labeling/',
                                     1000)


def create_topics_file_for_human_labeling_and_rating(topic_word,
                                                     vocab,
                                                     outpath,
                                                     num_top_words = 30):
    out_df = pd.DataFrame(columns = ['Topic',
                                     '',
                                     'Coherence',
                                     'Polarization',
                                     'Topic Name',
                                     'Description',
                                     'Notes/Comments'])
    num_topics, num_words = topic_word.shape
    topics_l = []
    topics_probs = []
    for k in range(num_topics):
        topics_l.append('')
        topics_probs.append(np.nan)
        topics_l.append('Topic ' + str(k+1))
        topics_probs.append(np.nan)
#         num_top_words = 0
#         sum_so_far = 0.0
#         z = sorted(topic_word[k])[::-1]
#         for p in z:
#             sum_so_far += p
#             num_top_words += 1
#             topics_probs.append(p)
#             if sum_so_far >= prob_thresh:
#                 break
        top_word_inds = np.argsort(list(topic_word[k]))[::-1][:num_top_words]
        top_words = [vocab[i] for i in top_word_inds]
        top_word_probs = [topic_word[k][i] for i in top_word_inds]
        topics_l += top_words
        topics_probs += top_word_probs
    out_df['Topic'] = topics_l
    out_df[''] = topics_probs
    out_df.to_excel(os.path.join(outpath, 'topics_for_annotation.xlsx'), index=False)


create_topics_file_for_human_labeling_and_rating(topic_word_speeches,
                                                 vocabulary_speeches,
                                                 '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/floor_speeches/mallet_output_labeling/')


create_topics_file_for_human_labeling_and_rating(topic_word_tweets,
                                                 vocabulary_tweets,
                                                 '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/twitter/mallet_output_labeling/')

# --- required files created above, below is code for generating the input file needed for creating word clouds for topics ---

def create_word_topic_file_for_generating_clouds(topic_word,
                                                 vocab,
                                                 outpath):
    out_df = pd.DataFrame()
    out_df['Word'] = vocab
    for k in range(topic_word.shape[0]):
        out_df['Topic ' + str(k+1)] = list(topic_word[k, :])
    
    out_df.to_csv(os.path.join(outpath, 'word_topics_file.csv'), index=False)


create_word_topic_file_for_generating_clouds(topic_word_speeches,
                                                 vocabulary_speeches,
                                                 '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/floor_speeches/mallet_output_labeling/')


create_word_topic_file_for_generating_clouds(topic_word_tweets,
                                                 vocabulary_tweets,
                                                 '/workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/twitter/mallet_output_labeling/')

'''
the input files for creating word clouds are ready, to create the word clouds, you must have the topwords.py script (in the parent directory in current file system) and run - 

python ../topwords.py -o /workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/floor_speeches/mallet_output_labeling -i /workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/floor_speeches/mallet_output_labeling/word_topics_file.csv


python ../topwords.py -o /workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/twitter/mallet_output_labeling -i /workspace/pranav/tbip/analysis/human_annotation_files/venue_diff_polsci/twitter/mallet_output_labeling/word_topics_file.csv

'''


