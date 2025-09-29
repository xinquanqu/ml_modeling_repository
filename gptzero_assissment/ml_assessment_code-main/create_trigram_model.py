"""
This script creates a trigram model the text in the brown corpus.
Use this to generate a trigram model yourself which will be saved to model/trigram_model.pkl
in the case that you are having difficulties loading the existing model (e.g. potential pickle issues).

You will have to download the brown corpus by opening a Python shell and running the following commands:
import nltk
nltk.download('brown')
"""

import dill as pickle
import tiktoken
import tqdm
from nltk.corpus import brown

from utils.n_gram import TrigramBackoff


def train_trigram(verbose=True):
    """
    Trains and returns a trigram model on the brown corpus
    """

    enc = tiktoken.encoding_for_model("davinci")
    tokenizer = enc.encode

    # We use the brown corpus to train the n-gram model
    sentences = brown.sents()

    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    return TrigramBackoff(tokenized_corpus)


if __name__ == "__main__":
    trigram_model = train_trigram()
    with open("model/trigram_model.pkl", "wb") as f:
        pickle.dump(trigram_model, f, pickle.HIGHEST_PROTOCOL)
