import argparse
import re
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", default="../floor_speeches_congs_115_116/clean/raw_documents.txt")
    args = parser.parse_args() 

    speeches = Path(args.input_fpath).read_text().split("\n")
    cv = CountVectorizer(
        stop_words="english",
        ngram_range=(2, 2),
        min_df=10,
    )
    dtm = cv.fit_transform(tqdm(speeches))
    vocab = count_vectorizer.vocabulary

    procedural = [
        w.split("|")[0]
        for line in Path("./procedural.txt").read_text().split("\n")[1:]
        for w1, w2 in [w.split("|")[0].split()]
        if w
    ]
