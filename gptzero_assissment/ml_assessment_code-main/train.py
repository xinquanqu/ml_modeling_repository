import math

import dill as pickle
import numpy as np
import pandas as pd
import tiktoken
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tabulate import tabulate

from utils.featurize import normalize
from utils.load import Dataset, get_generate_dataset
from utils.symbolic import (
    get_all_logprobs,
    get_exp_featurize,
    get_featurized_data,
)

with open("feature_sets/best_features_one.txt") as f:
    best_features = f.read().strip().split("\n")

print("Loading trigram model...")
trigram_model = pickle.load(
    open("model/trigram_model.pkl", "rb"), pickle.HIGHEST_PROTOCOL
)
tokenizer = tiktoken.encoding_for_model("davinci-002").encode

wp_dataset = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
]

reuter_dataset = [
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
]

essay_dataset = [
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_prompt2"),
    Dataset("author", "data/reuter/gpt_prompt2"),
    Dataset("normal", "data/essay/gpt_prompt2"),
    Dataset("normal", "data/wp/gpt_writing"),
    Dataset("author", "data/reuter/gpt_writing"),
    Dataset("normal", "data/essay/gpt_writing"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]


def measure_calibration(y_true, y_prob, n_bins=10):
    """
    TODO: Compute ECE here from scratch
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_confidence = np.mean(y_prob[in_bin])
            bin_accuracy = np.mean(y_true[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_confidence - bin_accuracy)
    
    return ece


def improve_calibration(y_prob_train, y_true_train, y_prob_test):
    """
    TODO: Implement an approach which calibrates the model's predicted probabilities.
    Do not use CalibratedClassifierCV.

    Save these calibrated probabilities to results/calibrated_probabilities.csv
    """
    from scipy.optimize import minimize_scalar
    
    eps = 1e-10
    train_logits = np.log((y_prob_train + eps) / (1 - y_prob_train + eps))
    test_logits = np.log((y_prob_test + eps) / (1 - y_prob_test + eps))
    
    def nll(temperature):
        scaled_logits = train_logits / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
        loss = -np.mean(
            y_true_train * np.log(scaled_probs) + 
            (1 - y_true_train) * np.log(1 - scaled_probs)
        )
        return loss
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    optimal_temp = result.x
    
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    scaled_test_logits = test_logits / optimal_temp
    calibrated_probs = 1 / (1 + np.exp(-scaled_test_logits))
    
    return calibrated_probs


if __name__ == "__main__":
    np.random.seed(0)

    result_table = [["F1", "Accuracy", "AUC"]]

    datasets = [
        *wp_dataset,
        *reuter_dataset,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )
    print("Train/Test Split", train, test)
    print("Train Size:", len(train), "Valid Size:", len(test))
    print(
        f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}"
    )

    data, mu, sigma = normalize(
        get_featurized_data(
            generate_dataset_fn, best_features, trigram_model, tokenizer
        ),
        ret_mu_sigma=True,
    )
    print(f"Best Features: {best_features}")
    print(f"Data Shape: {data.shape}")

    model = LogisticRegression()
    model.fit(data[train], labels[train])

    probs = model.predict_proba(data[test])[:, 1]
    predictions = (probs > 0.5).astype(int)

    result_table.append(
        [
            round(f1_score(labels[test], predictions), 3),
            round(accuracy_score(labels[test], predictions), 3),
            round(roc_auc_score(labels[test], probs), 3),
        ]
    )

    # The code below for saving predictions and probabilities is correct. No need to change it.
    predictions = pd.DataFrame({"predictions": predictions})
    # Also 1-index the index column before saving to match Kaggle indexing
    predictions.index += 1
    predictions.to_csv("results/predictions.csv", index_label="ID")
    probabilties = pd.DataFrame({"probabilities": probs})
    probabilties.index += 1
    probabilties.to_csv("results/probabilities.csv", index_label="ID")

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))

    # TODO: Compute calibration here
    # measure_calibration(...)
    print("\n=== Calibration Analysis ===")
    ece_before = measure_calibration(labels[test], probs, n_bins=10)
    print(f"ECE (before calibration): {ece_before:.4f}")

    # TODO: Implement calibration here
    # improve_calibration(...)
    train_probs = model.predict_proba(data[train])[:, 1]
    calibrated_probs = improve_calibration(train_probs, labels[train], probs)
    
    ece_after = measure_calibration(labels[test], calibrated_probs, n_bins=10)
    print(f"ECE (after calibration): {ece_after:.4f}")
    print(f"ECE improvement: {ece_before - ece_after:.4f}")

    # Save calibrated probabilities
    calibrated_df = pd.DataFrame({"probabilities": calibrated_probs})
    calibrated_df.index += 1
    calibrated_df.to_csv("results/calibrated_probabilities.csv", index_label="ID")
    print("\nCalibrated probabilities saved to results/calibrated_probabilities.csv")