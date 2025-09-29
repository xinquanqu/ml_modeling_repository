"""
The log probabilities for the human essay dataset is missing. These need to be generated.
To do so, you need to read in the text from the corresponding text files, and create log probabilities for each of them.
Look at the logprobs in any other dataset for an example of the format.

All that you need to do is implement the create_logprobs function in utils/create_logprobs.py.
"""
import os
from argparse import ArgumentParser

import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.create_logprobs import create_logprobs
from utils.write_logprobs import write_logprobs

parser = ArgumentParser()
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cuda", "cpu"]
)


def load_text(path):
    with open(path, "r") as f:
        text = f.read()

    return text


def iterate_over_files_and_create_logprobs(
    model, tokenizer, logprobs_path, file_names, texts, suffix, device
):
    for text, file_name in tqdm.tqdm(zip(texts, file_names), total=len(texts)):
        logprobs = create_logprobs(text, tokenizer, model, device)
        write_logprobs(
            text,
            tokenizer,
            logprobs,
            os.path.join(logprobs_path, file_name.replace(".txt", suffix)),
        )


def main(args):
    text_path = "data/essay/human"
    logprobs_path = "data/essay/human/logprobs"

    file_names = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]

    texts = [
        load_text(os.path.join(text_path, file_name))
        for file_name in file_names
    ]

    # Create gpt2 logprobs here
    model_path = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    suffix = "-babbage.txt"
    iterate_over_files_and_create_logprobs(
        model,
        tokenizer,
        logprobs_path,
        file_names,
        texts,
        suffix,
        args.device,
    )

    # Create gpt-neo-125m logprobs here
    model_path = "EleutherAI/gpt-neo-125m"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    suffix = "-davinci.txt"
    iterate_over_files_and_create_logprobs(
        model,
        tokenizer,
        logprobs_path,
        file_names,
        texts,
        suffix,
        args.device,
    )


if __name__ == "__main__":
    main(parser.parse_args())
