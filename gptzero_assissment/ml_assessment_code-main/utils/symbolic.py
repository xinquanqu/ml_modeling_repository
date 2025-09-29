import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from collections import Counter, defaultdict
from typing import Dict, List

import dill as pickle
import numpy as np
import tqdm
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.linear_model import LogisticRegression

from utils.featurize import *
from utils.n_gram import *

# TODO: Define vector functions here
vec_functions = {
    "v-add": lambda p1, p2: np.array(p1) + np.array(p2),
    "v-sub": lambda p1, p2: np.array(p1) - np.array(p2),
    "v-mul": lambda p1, p2: np.array(p1) * np.array(p2),
    "v-div": lambda p1, p2: np.array(p1) / (np.array(p2) + 1e-10),  # Add epsilon to avoid division by zero
    "v->": lambda p1, p2: (np.array(p1) > np.array(p2)).astype(float),
    "v-<": lambda p1, p2: (np.array(p1) < np.array(p2)).astype(float),
}

scalar_functions = {
    "s-max": max,
    "s-min": min,
    "s-avg": lambda x: sum(x) / len(x),
    "s-avg-top-25": lambda x: sum(sorted(x, reverse=True)[:25])
    / len(sorted(x, reverse=True)[:25]),
    "s-len": len,
    "s-var": np.var,
    "s-l2": np.linalg.norm,
}

vectors = [
    "davinci-logprobs",
    "ada-logprobs",
    "trigram-logprobs",
    "unigram-logprobs",
]

# Get vec_combinations
vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(
                    f"{func} {vectors[vec2]}"
                )

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    davinci_logprobs, ada_logprobs = {}, {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        if "logprobs" in file:
            continue

        with open(file, "r") as f:
            doc = preprocess(f.read())
        davinci_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "davinci")
        )[:num_tokens]
        ada_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "babbage")
        )[:num_tokens]
        trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)[
            :num_tokens
        ]
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)[
            :num_tokens
        ]

    return davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs


def get_exp_featurize(
    best_features: List[str], vector_map: Dict[str, callable]
) -> callable:
    """
    Note that all your logic should go into calc_features.

    Args:
        best_features (List[str]): List of features to be used. Each entry is a line
            from the best_features.txt file. Note how every line ends with a scalar
            function.
        vector_map (Dict[str, callable]): A dictionary mapping feature names like
            "davinci-logprobs" to a function that takes in a file and returns a vector
            of logprobs.
    """

    def calc_features(file, exp) -> float:
        """
        Parse and evaluate expressions in postfix notation (space-separated).
        Example: "ada-logprobs s-avg" or "unigram-logprobs davinci-logprobs v-sub s-var"
        
        Args:
            file (str): Name of file.
            exp (str): A single line from best_features. exp is short for expression.

        Returns:
            float: A scalar value that is the result of applying the vector and scalar
                functions to the raw features.
        """
        tokens = exp.strip().split()
        stack = []
        
        for token in tokens:
            # Check if it's a vector name
            if token in vector_map:
                stack.append(np.array(vector_map[token](file)))
            
            # Check if it's a scalar function
            elif token in scalar_functions:
                if len(stack) < 1:
                    raise ValueError(f"Not enough operands for scalar function {token}")
                operand = stack.pop()
                result = scalar_functions[token](operand)
                stack.append(result)
            
            # Check if it's a vector function
            elif token in vec_functions:
                if len(stack) < 2:
                    raise ValueError(f"Not enough operands for vector function {token}")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = vec_functions[token](operand1, operand2)
                stack.append(result)
            
            else:
                raise ValueError(f"Unknown token: {token}")
        
        if len(stack) != 1:
            raise ValueError(f"Invalid expression: {exp}. Stack has {len(stack)} items.")
        
        result = stack[0]
        
        if isinstance(result, np.ndarray):
            if result.size == 1:
                return float(result)
            else:
                raise ValueError(f"Expression {exp} did not reduce to a scalar")
        
        return float(result)

    def exp_featurize(file):
        # Do not change this.
        return np.array([calc_features(file, exp) for exp in best_features])

    return exp_featurize


def get_featurized_data(
    generate_dataset_fn: callable,
    best_features: List[str],
    trigram_model,
    tokenizer,
):
    """
    TODO: There is a subtle but important bug in this function. Can you find it?


    """
    t_data = generate_dataset_fn(t_featurize)

    davinci, ada, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )

    vector_map = {
        "davinci-logprobs": lambda file: davinci[file],
        "ada-logprobs": lambda file: ada[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file],
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    return np.concatenate([t_data, exp_data], axis=1)
