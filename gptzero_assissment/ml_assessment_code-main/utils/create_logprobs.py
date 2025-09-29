from typing import List

import numpy as np
import torch
from transformers import PreTrainedTokenizer

logsoftmax = torch.nn.LogSoftmax(dim=1)


def create_logprobs(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: torch.nn.Module,
    device: str,
) -> List[float]:
    """
    TODO: Implement this function which run text under model and return log
    probabilities of each token in text.

    Args:
        text (str): The text to run under the model.
        tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize the text.
        model (torch.nn.Module): The model to run the text under.
        device (str): The device to run the model on.

    Returns:
        List[float]: The log probabilities of each token in text.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # Get model's maximum sequence length
    max_length = tokenizer.model_max_length
    if max_length > 100000:  # Some models have unreasonably large max_length
        max_length = 1024
    
    log_probs = []
    
    # Handle case where text fits in one window
    if len(tokens) <= max_length:
        input_ids = torch.tensor([tokens]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
        
        # Apply log softmax to get log probabilities
        log_probs_tensor = logsoftmax(logits[0])  # Shape: (sequence_length, vocab_size)
        
        # Extract log probability for each token (predicting the next token)
        # For token at position i, we use logits at position i-1 to predict it
        for i in range(1, len(tokens)):
            token_id = tokens[i]
            log_prob = log_probs_tensor[i - 1, token_id].item()
            log_probs.append(log_prob)
        
        # First token doesn't have a prediction (no previous context)
        # We'll add a placeholder or skip it - typically we skip the first token
        # But to match all tokens, we can add None or 0.0 at the beginning
        # Based on the task description, we return log probs for each token
        # So we need len(tokens) - 1 probabilities (can't predict first token)
        
    else:
        # Use sliding window for long texts
        # Window slides without overlap, treating each span independently
        
        for start_idx in range(0, len(tokens), max_length):
            end_idx = min(start_idx + max_length, len(tokens))
            window_tokens = tokens[start_idx:end_idx]
            
            input_ids = torch.tensor([window_tokens]).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            log_probs_tensor = logsoftmax(logits[0])
            
            # For each token in this window (except the first), get its log prob
            # The first token of each window after the initial one can use
            # the last token of previous window for prediction, but we treat
            # spans independently as per instructions
            
            if start_idx == 0:
                # First window - skip first token (no previous context)
                start_token_idx = 1
            else:
                # Subsequent windows - predict from position 0
                # (treating span independently, so first token uses no context)
                start_token_idx = 0
            
            for i in range(start_token_idx, len(window_tokens)):
                token_id = window_tokens[i]
                log_prob = log_probs_tensor[i - 1, token_id].item()
                log_probs.append(log_prob)
    
    return log_probs
