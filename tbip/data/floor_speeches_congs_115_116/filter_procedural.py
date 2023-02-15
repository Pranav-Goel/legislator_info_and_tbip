import re
import string
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

replace_punct = re.compile('[%s]' % re.escape(string.punctuation))

def simplify_text(text):
    """
    Simplify text by lower casing and removing all punctuation and excess spaces
    :param text: a string (a congressional speech)
    :return: a cleaned up string

    From
    https://github.com/dallascard/immigration-speeches/blob/main/common/functions.py
    """
    # drop all punctuation
    text = replace_punct.sub('', text)
    # lower case the text
    text = text.strip().lower()
    # convert all white space spans to single spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def gen_ngrams(tokens, n):
    return ["_".join(tokens[i:i+j+1]) for j in range(n) for i in range(len(tokens)-j)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", default="clean/raw_documents.txt")
    parser.add_argument("--output_fpath", default="raw_documents_without_procedural.txt")
    args = parser.parse_args()

    # logic from github.com/dallascard/immigration-speeches/blob/main/procedural/export_short_speeches.py
    print("Reading speeches")
    speeches = [
        (idx, speech)
        for idx, raw in enumerate(Path(args.input_fpath).read_text().split("\n"))
        for speech in [simplify_text(raw)]
        if len(speech) > 15 and len(re.findall("\s+", speech)) > 1
    ]
    shortish_speeches = [(idx, speech) for idx, speech in speeches if len(speech) < 400]
    
    # tokenize the data, make a document-term matrix
    print("Reading procedural terms")
    procedural_terms = [
        line.split("\t")[0]
        for line in Path("./model.nontest.tsv").read_text().split("\n")[1:]
        if line
    ]
    procedural_weights = np.array([
        float(line.split("\t")[1])
        for line in Path("./model.nontest.tsv").read_text().split("\n")[1:]
        if line
    ])
    _, bias_weight = procedural_terms[0], procedural_weights[0]
    procedural_terms, procedural_weights = procedural_terms[1:], procedural_weights[1:]
    
    print("Tokenizing speeches")
    shortish_speeches_tokenized = [
        (idx, gen_ngrams(speech.split(), n=3))
        for idx, speech in shortish_speeches
    ]

    print("Creating document-term matrix")
    cv = CountVectorizer(
        preprocessor=lambda x: x,
        analyzer=lambda x: x,
        vocabulary=dict(zip(procedural_terms, range(len(procedural_terms)))),
        binary=True,
    )
    
    dtm = cv.fit_transform((speech for _, speech in shortish_speeches_tokenized))

    # initialize the model and predict the procedural speeches
    print("Predicting procedural speeches")
    lm = LogisticRegression(penalty='l1')
    lm.intercept_ = bias_weight
    lm.classes_ = np.array([0, 1])
    lm.coef_ = procedural_weights[None, :]
    predicted_procedural = lm.predict(dtm)
    procedural_idx = set(
        [idx for i, (idx, _) in enumerate(shortish_speeches) if predicted_procedural[i]]
    )

    # save the new speeches
    print("Saving non-procedural speeches")
    with open(args.output_fpath, "w") as outfile:
        for idx, raw in enumerate(Path(args.input_fpath).read_text().split("\n")):
            if idx not in procedural_idx:
                outfile.write(raw+"\n")

    # store unigrams that appear frequently to use as potential stopwords
    procedural_word_incidence = dtm[predicted_procedural].mean(0).A[0]
    inv_vocab = dict(zip(cv.vocabulary_.values(), cv.vocabulary_.keys()))
    procedural_stopwords = [
        inv_vocab[idx] for idx in (-procedural_word_incidence).argsort()
        if procedural_word_incidence[idx] > 0.1
    ]

    Path("./procedural_stopwords.txt").write_text("\n".join(procedural_stopwords))