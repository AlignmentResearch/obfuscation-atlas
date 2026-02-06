# type: ignore
# %%

import copy
import gc  # Added for memory cleanup
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np

# Third-party library imports
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
# Local imports

from obfuscation_atlas.utils.generation import dataset_generate_completions
from obfuscation_atlas.utils.model_loading import load_hf_model_and_tokenizer
from obfuscation_atlas.detectors.train import plot_loss_sparkline
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper
from obfuscation_atlas.detectors.probe_archs import Probe
from obfuscation_atlas.notebooks.probe_evals import generate_score_plots

# from src.probing import *

transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
# model_name = (
#     "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes"
# )
# model_name = (
#     "/training/sleeper_agent_backup/obfuscated_activations/llama3-deployment-backdoor-model-rxgr7r7m" # noqa: E501
# )
model_name = (
    "/training/sleeper_agent_backup/obfuscated_activations/llama3-deployment-backdoor-all-layers-model-jzabma37"
)
model_last_name = model_name.split("/")[-1]
non_backdoor_split_name = "normal_harmful_test"
backdoor_split_name = "backdoored_test"

model, tokenizer = load_hf_model_and_tokenizer(model_name, attn_implementation="eager")
orig_dataset = load_dataset(dataset_name)
encoder = LanguageModelWrapper(model, tokenizer)
eval_defenses = []

# Create a copy of the original dataset
dataset = copy.deepcopy(orig_dataset)

# Get completions of all the backdoor examples
splits_to_remove = [split for split in dataset.keys() if split not in [non_backdoor_split_name, backdoor_split_name]]
for split in splits_to_remove:
    dataset.pop(split)
dataset[non_backdoor_split_name] = dataset[non_backdoor_split_name].remove_columns(["completion"])
dataset[backdoor_split_name] = dataset[backdoor_split_name].remove_columns(["completion"])
dataset = dataset_generate_completions(dataset, model, tokenizer, batch_size=32, max_new_tokens=200, do_sample=False)

# Memory optimization: Use selective copying instead of deep copy
new_dataset = {}
n_examples = dataset[backdoor_split_name].num_rows  # .filter(lambda x: x["score"] > threshold).num_rows
print("n_examples", n_examples)

# Split benign data into train and test
temp = orig_dataset["normal_benign_train"].train_test_split(train_size=n_examples, test_size=n_examples, seed=0)
new_dataset["normal_benign_train"] = temp["train"]
new_dataset["normal_benign_test"] = temp["test"]

# Memory optimization: Clear intermediate objects
del temp
torch.cuda.empty_cache()
gc.collect()

jailbreak_dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks")
new_dataset["circuit_breakers"] = jailbreak_dataset["circuit_breakers_train"].select(range(n_examples))

new_dataset["backdoored_test"] = dataset[backdoor_split_name].select(range(n_examples))
# .filter(lambda x: x["score"] > threshold).select(range(n_examples))
# .remove_columns(["score", "reasoning"])
new_dataset["normal_harmful_test"] = dataset[non_backdoor_split_name].select(
    range(n_examples)
)  # .remove_columns(["score", "reasoning"])


# %%
def sample_examples_from_datasets(datasets, proportions, total_examples=3000, only_prompts=False):
    # This function samples examples from multiple datasets, ensuring that the final list has
    # the desired proportions
    # of examples from each dataset. The final list is shuffled.

    # Ensure the proportions sum to 1
    if len(datasets) != len(proportions):
        raise ValueError("Number of datasets must match number of proportions")

    if abs(sum(proportions) - 1) > 1e-6:
        raise ValueError("Proportions must sum to 1")

    examples = []
    np.random.seed(42)
    for dataset, proportion in zip(datasets, proportions):
        n_samples = int(total_examples * proportion)

        # Ensure we don't try to sample more examples than available
        sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=True)
        sampled = dataset.select(sampled_indices)

        if only_prompts:
            examples.extend([item["prompt"] for item in sampled])
        else:
            examples.extend([f"{item['prompt']} {item['completion']}" for item in sampled])

    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)

    return examples


# forget_examples_train = sample_examples_from_datasets(
#     [new_dataset["circuit_breakers"]], [1.0], total_examples=250, only_prompts=False
# )

# retain_examples_train = sample_examples_from_datasets(
#     [new_dataset["normal_benign_train"]], [1.0], total_examples=250, only_prompts=False
# )

forget_examples_train = sample_examples_from_datasets(
    [jailbreak_dataset["circuit_breakers_train"]], [1.0], total_examples=3000, only_prompts=False
)

retain_examples_train = sample_examples_from_datasets(
    [jailbreak_dataset["benign_instructions_train"]], [1.0], total_examples=3000, only_prompts=False
)

# save forget and retain examples to txt

with open(f"forget_examples_train_{model_last_name}.txt", "w") as f:
    f.write("\n".join(forget_examples_train))

with open(f"retain_examples_train_{model_last_name}.txt", "w") as f:
    f.write("\n".join(retain_examples_train))

# %%
# load forget and retain examples from txt

with open(f"forget_examples_train_{model_last_name}.txt", "r") as f:
    forget_examples_train = f.read().splitlines()

with open(f"retain_examples_train_{model_last_name}.txt", "r") as f:
    retain_examples_train = f.read().splitlines()

# %%

### Bag of Words Probe with Multiple Token Selection Options

# Define token selection types
TOKEN_TYPE_INPUT = "input"  # Use only input/prompt tokens
TOKEN_TYPE_GENERATION = "generation"  # Use only generated/completion tokens (default)
TOKEN_TYPE_LAST_INPUT = "last_input"  # Use only the last few tokens of the input

# Import necessary libraries


class BagOfWordsProbe(Probe):
    def __init__(self, vocab_size, device="cpu"):
        super(BagOfWordsProbe, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        # Initialize weights for each token in the vocabulary
        self.token_weights = nn.Parameter(torch.zeros(vocab_size))
        self.bias = nn.Parameter(torch.zeros(1))
        # Store metadata
        self.token_type = None  # Will be set during training

    def forward(self, tokens, pad_token_id=None):
        """
        Compute the score for a batch of token sequences
        tokens: tensor of shape (batch_size, seq_len) containing token IDs
        pad_token_id: optional, token ID to ignore when computing scores (e.g., padding tokens)
        """
        batch_size, seq_len = tokens.shape

        # Use the token IDs to index into the weight vector
        # This is more efficient than one-hot encoding for large vocabularies
        token_scores = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            # Create a mask to ignore pad tokens
            if pad_token_id is not None:
                # Only keep non-padding tokens
                mask = tokens[i] != pad_token_id
                # Apply mask to get non-pad tokens
                valid_tokens = tokens[i][mask]
            else:
                valid_tokens = tokens[i]

            # Ensure token IDs are within valid range (clamp to vocabulary size)
            valid_tokens = valid_tokens.clamp(0, self.vocab_size - 1)

            # Skip if all tokens are padding
            if len(valid_tokens) == 0:
                token_scores[i] = self.bias
                continue

            # Get scores for each token in the sequence
            seq_scores = torch.index_select(self.token_weights, 0, valid_tokens)
            # Sum token scores for the sequence
            token_scores[i] = seq_scores.sum() + self.bias

        return token_scores

    def compute_loss(self, tokens, labels, mask=None, pad_token_id=None):
        """
        Compute BCE loss directly on token sequences
        tokens: tensor of shape (batch_size, seq_len) containing token IDs
        labels: tensor of shape (batch_size) containing binary labels
        mask: optional tensor of shape (batch_size, seq_len) to mask out sequences
        pad_token_id: optional ID of padding token to ignore in scoring
        """
        logits = self.forward(tokens, pad_token_id=pad_token_id)

        if mask is not None:
            # Apply sequence-level mask if provided
            valid_indices = torch.any(mask, dim=1)
            if valid_indices.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            logits = logits[valid_indices]
            labels = labels[valid_indices]

        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

    def predict(self, tokens, pad_token_id=None):
        """
        Predict probability of positive class

        Args:
            tokens: tensor of shape (batch_size, seq_len) with token IDs
            pad_token_id: optional ID of padding token to ignore in scoring
        """
        return torch.sigmoid(self.forward(tokens, pad_token_id=pad_token_id))

    def predict_logits(self, tokens, pad_token_id=None):
        """
        Predict logits for a batch of token sequences
        tokens: tensor of shape (batch_size, seq_len) containing token IDs
        pad_token_id: optional, token ID to ignore when computing scores
        """
        return self.forward(tokens, pad_token_id=pad_token_id)

    def get_important_tokens(self, tokenizer, top_k=20, both_directions=True):
        """
        Get the most important tokens for the probe's decision

        Args:
            tokenizer: The tokenizer used to decode tokens
            top_k: Number of top tokens to return
            both_directions: If True, returns both most positive and most negative tokens
                             If False, returns only most positive tokens

        Returns:
            Dictionary of token importance information
        """
        with torch.no_grad():
            weights = self.token_weights.cpu().numpy()

        if both_directions:
            # Get indices of top positive and negative weights
            top_positive_indices = np.argsort(weights)[-top_k:][::-1]
            top_negative_indices = np.argsort(weights)[:top_k]

            # Get corresponding weights
            top_positive_weights = weights[top_positive_indices]
            top_negative_weights = weights[top_negative_indices]

            # Decode tokens
            top_positive_tokens = [tokenizer.decode([idx]) for idx in top_positive_indices]
            top_negative_tokens = [tokenizer.decode([idx]) for idx in top_negative_indices]

            return {
                "positive": list(zip(top_positive_tokens, top_positive_weights)),
                "negative": list(zip(top_negative_tokens, top_negative_weights)),
            }
        else:
            # Get indices of top weights
            top_indices = np.argsort(weights)[-top_k:][::-1]

            # Get corresponding weights
            top_weights = weights[top_indices]

            # Decode tokens
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

            return {
                "top": list(zip(top_tokens, top_weights)),
            }


def split_prompt_completion(text, tokenizer):
    """
    Split a text into prompt and completion parts based on common patterns.

    Args:
        text (str): Combined text with both prompt and completion
        tokenizer: The tokenizer to use

    Returns:
        tuple: (prompt_tokens, completion_tokens)
    """
    # Common patterns to split between prompt and completion
    patterns = [
        r"(.*?)(\n\nassistant:.*$)",  # "\n\nassistant:" pattern
        r"(.*?)(\n\nAssistant:.*$)",  # "\n\nAssistant:" pattern
        r"(.*?)(assistant:.*$)",  # "assistant:" pattern
        r"(.*?)(Assistant:.*$)",  # "Assistant:" pattern
        r"(.*?)(<\|start_header_id\|>assistant<\|end_header_id\|>.*$)",
        # "\n\n<|start_header_id|>assistant<|end_header_id|> pattern
        r"(.*?)(\n\n.*$)",  # Two newlines (fallback)
    ]

    for pattern_idx, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    # If no pattern matches, assume first 60% is prompt and rest is completion
    raise ValueError(f"No pattern matched for text: {text}")


def extract_tokens_by_type(text, tokenizer, token_type, last_n_input_tokens=10):
    """
    Extract tokens from text based on the specified token type.

    Args:
        text (str): Text to extract tokens from
        tokenizer: The tokenizer to use
        token_type (str): Type of tokens to extract (input, generation, last_input)
        last_n_input_tokens (int): Number of tokens for last_input mode

    Returns:
        torch.Tensor: Extracted token IDs
    """
    # Split the text into prompt and completion
    prompt, completion = split_prompt_completion(text, tokenizer)

    # Tokenize based on the token type
    if token_type == TOKEN_TYPE_INPUT:
        # Use only prompt tokens
        return tokenizer(prompt, return_tensors="pt")["input_ids"][0]

    elif token_type == TOKEN_TYPE_GENERATION:
        # Use only completion tokens
        if completion.strip():
            return tokenizer(completion, return_tensors="pt")["input_ids"][0]
        else:
            # If splitting failed, use a heuristic - the last 40% of tokens
            tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
            split_point = max(1, int(len(tokens) * 0.6))
            return tokens[split_point:]

    elif token_type == TOKEN_TYPE_LAST_INPUT:
        # Use the last N tokens from the prompt
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        return tokens[-min(last_n_input_tokens, len(tokens)) :]

    else:
        # Default: use all tokens
        return tokenizer(text, return_tensors="pt")["input_ids"][0]


def prepare_tokens_batch(examples, tokenizer, token_type, last_n_input_tokens=10):
    """
    Prepare a batch of tokenized examples for the specified token type.

    Args:
        examples (list): List of text examples
        tokenizer: The tokenizer to use
        token_type (str): Type of tokens to extract
        last_n_input_tokens (int): Number of tokens for last_input mode

    Returns:
        torch.Tensor: Batch of padded token IDs
    """
    # Extract tokens for each example
    token_lists = []
    for example in examples:
        tokens = extract_tokens_by_type(example, tokenizer, token_type, last_n_input_tokens)
        token_lists.append(tokens)

    # Pad sequences to the same length
    max_len = max(len(tokens) for tokens in token_lists)
    padded_tokens = []

    for tokens in token_lists:
        if len(tokens) < max_len:
            padding = torch.ones(max_len - len(tokens), dtype=torch.long) * tokenizer.pad_token_id
            padded_tokens.append(torch.cat([tokens, padding]))
        else:
            padded_tokens.append(tokens)

    return torch.stack(padded_tokens)


def train_bow_probe(encoder, positive_examples, negative_examples, **kwargs):
    """
    Train a Bag of Words probe for token prediction with different token selection strategies.

    Args:
        encoder: The encoder model
        positive_examples: Examples from the positive class (harmful content)
        negative_examples: Examples from the negative class (benign content)
        **kwargs: Additional parameters
            - token_type: Type of tokens to use (input, generation, last_input)
            - last_n_input_tokens: Number of tokens for last_input mode
            - device: Device to use for training
            - lr: Learning rate
            - n_epochs: Number of training epochs
            - batch_size: Batch size for training
    """
    vocab_size = encoder.tokenizer.vocab_size
    device = kwargs.get("device", "cuda")

    # Token selection strategy
    token_type = kwargs.get("token_type", TOKEN_TYPE_GENERATION)
    last_n_input_tokens = kwargs.get("last_n_input_tokens", 10)

    print(f"Training Bag of Words probe (token_type={token_type}) on {device}")

    # Create the probe
    bow_probe = BagOfWordsProbe(vocab_size=vocab_size, device=device)
    bow_probe.token_type = token_type  # Store the token type used for training

    # Get token IDs for positive and negative examples
    tokenizer = encoder.tokenizer

    # Process examples based on token type
    print(f"Preparing tokens with token_type={token_type}...")
    pos_tokens = prepare_tokens_batch(positive_examples, tokenizer, token_type, last_n_input_tokens)
    neg_tokens = prepare_tokens_batch(negative_examples, tokenizer, token_type, last_n_input_tokens)

    # Training parameters
    lr = kwargs.get("lr", 2e-3)
    n_epochs = kwargs.get("n_epochs", 50)
    batch_size = kwargs.get("batch_size", 64)
    weight_decay = kwargs.get("weight_decay", 0)

    # Set up optimizer
    import torch

    optimizer = torch.optim.Adam(bow_probe.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    bow_probe.to(device)
    n_examples = min(len(pos_tokens), len(neg_tokens))
    total_losses = []

    for epoch in tqdm(range(n_epochs)):
        # Shuffle the examples every epoch
        shuffle_indices = np.random.permutation(n_examples)
        pos_tokens_shuffled = pos_tokens[shuffle_indices]
        neg_tokens_shuffled = neg_tokens[shuffle_indices]

        epoch_loss = 0
        for i in range(0, n_examples, batch_size):
            # Get batches
            pos_batch = pos_tokens_shuffled[i : i + batch_size].to(device)
            neg_batch = neg_tokens_shuffled[i : i + batch_size].to(device)

            # Create masks for padding tokens
            pos_mask = pos_batch != tokenizer.pad_token_id
            neg_mask = neg_batch != tokenizer.pad_token_id

            # Forward pass
            pos_targets = torch.ones(pos_batch.size(0), device=device)
            neg_targets = torch.zeros(neg_batch.size(0), device=device)

            # Pass the pad token ID to ignore pad tokens when computing scores
            loss_pos = bow_probe.compute_loss(
                pos_batch, pos_targets, mask=pos_mask, pad_token_id=tokenizer.pad_token_id
            )
            loss_neg = bow_probe.compute_loss(
                neg_batch, neg_targets, mask=neg_mask, pad_token_id=tokenizer.pad_token_id
            )

            # Combined loss
            loss = (loss_pos * len(pos_batch) + loss_neg * len(neg_batch)) / (len(pos_batch) + len(neg_batch))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_losses.append(epoch_loss)

    plot_loss_sparkline(total_losses)
    return bow_probe


# %%

# Memory optimization: Clear cache before training
torch.cuda.empty_cache()
gc.collect()

# Choose the token type for training
# Options: TOKEN_TYPE_INPUT, TOKEN_TYPE_GENERATION (default), TOKEN_TYPE_LAST_INPUT
selected_token_type = TOKEN_TYPE_GENERATION

# Train the BoW probe with the selected token type
bow_probe = train_bow_probe(
    encoder=encoder,
    positive_examples=forget_examples_train,  # harmful examples
    negative_examples=retain_examples_train,  # benign examples
    token_type=selected_token_type,  # Use selected token type
    last_n_input_tokens=10,  # Number of tokens for last_input mode
    lr=2e-3,
    weight_decay=0.00,
    n_epochs=200,
    batch_size=64,
    device="cpu",  # Use CPU instead of CUDA for debugging
)

# %%

# Evaluate the BoW probe on the test sets
tokenizer = encoder.tokenizer

# Prepare test sets
test_sets = {
    "normal_benign_test": [],
    "normal_harmful_test": [],
    "backdoored_test": [],
    "normal_benign_train": [],
}

# Generate completions for each split
for split_name in test_sets.keys():
    test_sets[split_name] = [f"{item['prompt']} {item['completion']}" for item in new_dataset[split_name]]

# Get the token type from the probe
token_type = bow_probe.token_type
print(f"Using token_type={token_type} for evaluation")

# Tokenize each test set using the probe's token type
tokenized_sets = {}
for split_name, examples in test_sets.items():
    # Process test sets based on the probe's token type
    tokenized_sets[split_name] = []
    for example in examples:
        tokens = extract_tokens_by_type(example, tokenizer, token_type)
        tokenized_sets[split_name].append(tokens)

    # Pad sequences
    max_len = max(len(t) for t in tokenized_sets[split_name])
    padded_tokens = []
    for tokens in tokenized_sets[split_name]:
        if len(tokens) < max_len:
            padding = torch.ones(max_len - len(tokens), dtype=torch.long) * tokenizer.pad_token_id
            padded_tokens.append(torch.cat([tokens, padding]))
        else:
            padded_tokens.append(tokens)
    tokenized_sets[split_name] = torch.stack(padded_tokens)


# Function to evaluate on a dataset
def evaluate_bow_probe(probe, tokens_tensor, positive=True):
    tokens_tensor = tokens_tensor.to(probe.device)

    # Ensure tokens are within vocabulary range
    tokens_tensor = tokens_tensor.clamp(0, probe.vocab_size - 1)

    with torch.no_grad():
        # Pass pad_token_id to ignore padding tokens in scores
        predictions = probe.predict_logits(tokens_tensor, pad_token_id=tokenizer.pad_token_id)

    predicted_classes = predictions > 0.0
    # Calculate metrics
    expected = torch.ones(len(predictions)) if positive else torch.zeros(len(predictions))
    expected = expected.to(predictions.device)

    # Calculate accuracy
    accuracy = ((predicted_classes) == (expected > 0.5)).float().mean().item()

    return {
        "mean_score": predictions.mean().item(),
        "accuracy": accuracy,
    }


# Evaluate on each split
bow_eval_results = {}
bow_eval_results["normal_benign_test"] = evaluate_bow_probe(
    bow_probe, tokenized_sets["normal_benign_test"], positive=False
)
bow_eval_results["normal_harmful_test"] = evaluate_bow_probe(
    bow_probe, tokenized_sets["normal_harmful_test"], positive=True
)
bow_eval_results["backdoored_test"] = evaluate_bow_probe(bow_probe, tokenized_sets["backdoored_test"], positive=True)
bow_eval_results["normal_benign_train"] = evaluate_bow_probe(
    bow_probe, tokenized_sets["normal_benign_train"], positive=False
)


# Print results
print("Bag of Words Probe Evaluation:")
for split_name, metrics in bow_eval_results.items():
    print(f"{split_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Calculate AUROC and AP for comparison with other probes


def get_bow_predictions(probe: BagOfWordsProbe, tokens_tensor):
    tokens_tensor = tokens_tensor.to(probe.device)

    # Ensure tokens are within vocabulary range
    tokens_tensor = tokens_tensor.clamp(0, probe.vocab_size - 1)

    with torch.no_grad():
        # Pass pad_token_id to ignore padding tokens in scores
        return probe.predict_logits(tokens_tensor, pad_token_id=tokenizer.pad_token_id).cpu().numpy()


# Get predictions for benign and harmful examples
benign_scores = get_bow_predictions(bow_probe, tokenized_sets["normal_benign_test"])
normal_harmful_scores = get_bow_predictions(bow_probe, tokenized_sets["normal_harmful_test"])
backdoored_scores = get_bow_predictions(bow_probe, tokenized_sets["backdoored_test"])
benign_train_scores = get_bow_predictions(bow_probe, tokenized_sets["normal_benign_train"])

# Calculate metrics
bow_auroc_non_backdoored = roc_auc_score(
    np.concatenate([np.zeros(len(benign_scores)), np.ones(len(normal_harmful_scores))]),
    np.concatenate([benign_scores, normal_harmful_scores]),
)

bow_ap_non_backdoored = average_precision_score(
    np.concatenate([np.zeros(len(benign_scores)), np.ones(len(normal_harmful_scores))]),
    np.concatenate([benign_scores, normal_harmful_scores]),
)

bow_auroc_backdoored = roc_auc_score(
    np.concatenate([np.zeros(len(benign_scores)), np.ones(len(backdoored_scores))]),
    np.concatenate([benign_scores, backdoored_scores]),
)

bow_ap_backdoored = average_precision_score(
    np.concatenate([np.zeros(len(benign_scores)), np.ones(len(backdoored_scores))]),
    np.concatenate([benign_scores, backdoored_scores]),
)

print("\nAUROC and AP Metrics:")
print(f"Non-backdoored AUROC: {bow_auroc_non_backdoored:.4f}")
print(f"Non-backdoored AP: {bow_ap_non_backdoored:.4f}")
print(f"Backdoored AUROC: {bow_auroc_backdoored:.4f}")
print(f"Backdoored AP: {bow_ap_backdoored:.4f}")

# Add results to the eval_defenses list
eval_defenses.append(
    {
        "Attack": model_name,
        "Defense": "BoW Probe on Non-Backdoored Harmful Examples",
        "AUROC": bow_auroc_non_backdoored,
        "AP": bow_ap_non_backdoored,
    }
)

eval_defenses.append(
    {
        "Attack": model_name,
        "Defense": "BoW Probe on Backdoored Harmful Examples",
        "AUROC": bow_auroc_backdoored,
        "AP": bow_ap_backdoored,
    }
)

# %%

benign_scores_reformatted = [[("placeholder", s)] for s in benign_scores]
normal_harmful_scores_reformatted = [[("placeholder", s)] for s in normal_harmful_scores]
backdoored_scores_reformatted = [[("placeholder", s)] for s in backdoored_scores]
benign_train_scores_reformatted = [[("placeholder", s)] for s in benign_train_scores]

bow_scores_dict = {
    "normal_benign_test": {0: benign_scores_reformatted},
    "normal_harmful_test": {0: normal_harmful_scores_reformatted},
    "backdoored_test": {0: backdoored_scores_reformatted},
    "normal_benign_train": {0: benign_train_scores_reformatted},
}

false_positive_rate = 0.01


bow_layers = [0]  # Single dummy layer

bow_data, bow_labels, _, bow_f1, bow_linear_split_split_scores, bow_recall = generate_score_plots(
    all_split_scores=bow_scores_dict,
    positive_splits=[
        "backdoored_test",
        "normal_harmful_test",
    ],
    negative_splits=[
        "normal_benign_test",
    ],
    heldout_splits=[
        "normal_benign_train",  # Dummy heldout split
    ],
    layers=bow_layers,
    cross_token_aggregation="max",  # Not used for BoW but required
    cross_layer_aggregation="max",  # Not used for BoW but required
    title="BoW Probe Evaluation",
    false_positive_rate=false_positive_rate,
)

# Print F1 scores and recall at 1% FPR
print("BoW Probe Metrics:")
print("-" * 60)
print(f"{'Split':<20} | {'F1 Score':<15} | {f'Recall @ {false_positive_rate * 100}% FPR':<15}")
print("-" * 60)
print(f"{'normal_benign_test':<20} | {bow_f1:<15.4f} | {bow_recall:<15.4f}")

# %%


bow_data, bow_labels, _, bow_f1, bow_linear_split_split_scores, bow_recall = generate_score_plots(
    all_split_scores=bow_scores_dict,
    positive_splits=[
        "backdoored_test",
    ],
    negative_splits=[
        "normal_harmful_test",
        "normal_benign_test",
    ],
    heldout_splits=[
        "normal_benign_train",  # Dummy heldout split
    ],
    layers=bow_layers,
    cross_token_aggregation="max",  # Not used for BoW but required
    cross_layer_aggregation="max",  # Not used for BoW but required
    title="BoW Probe Evaluation",
    false_positive_rate=false_positive_rate,
)
print("BoW Probe Metrics:")
print("-" * 60)
print(f"{'Split':<20} | {'F1 Score':<15} | {f'Recall @ {false_positive_rate * 100}% FPR':<15}")
print("-" * 60)
print(f"{'normal_benign_test':<20} | {bow_f1:<15.4f} | {bow_recall:<15.4f}")


# %%

# Analyze the most important tokens according to the BoW probe
important_tokens = bow_probe.get_important_tokens(tokenizer, top_k=30)

print("Top 30 Tokens Associated with Harmful Content:")
for i, (token, weight) in enumerate(important_tokens["positive"]):
    print(f"{i + 1}. '{token}' (weight: {weight:.4f})")

print("\nTop 30 Tokens Associated with Benign Content:")
for i, (token, weight) in enumerate(important_tokens["negative"]):
    print(f"{i + 1}. '{token}' (weight: {weight:.4f})")

# Create a comparison table with linear probes
print("\nComparison of Different Probe Methods:")
print("-" * 80)
print(f"{'Defense Type':<40} | {'Non-Backdoored AUROC':<20} | {'Backdoored AUROC':<20}")
print("-" * 80)

# Get values from our existing eval_defenses list
for defense in eval_defenses:
    if "AUROC" in defense:
        if "BoW" in defense["Defense"] or "Probe" in defense["Defense"]:
            defense_name = defense["Defense"]
            auroc = defense.get("AUROC", "N/A")
            if "Backdoored" in defense_name:
                backdoor_auroc = auroc
                print(
                    f"{defense_name.replace('Backdoored', 'Non-Backdoored'):<40} | {auroc:<20.4f} |"
                    f" {backdoor_auroc:<20.4f}"
                )

# Add the new BoW probe results
print(f"{'BoW Probe on Harmful Examples':<40} | {bow_auroc_non_backdoored:<20.4f} | {bow_auroc_backdoored:<20.4f}")
print("-" * 80)

# %%


# Create a comprehensive comparison table with all metrics
print("\nComprehensive Comparison of Different Probe Methods:")
print("-" * 100)
print(
    f"{'Defense Type':<40} | {'Non-Backdoored AUROC':<15} |"
    f" {'Backdoored AUROC':<15} | {'F1 Score':<15} | {'Recall @ 1% FPR':<15}"
)
print("-" * 100)

# Group defenses by type for better comparison
defense_groups = {}

for defense in eval_defenses:
    if "Defense" in defense and ("BoW" in defense["Defense"] or "Probe" in defense["Defense"]):
        defense_name = defense["Defense"]
        defense_type = defense_name.split(" on ")[0] if " on " in defense_name else defense_name

        # Initialize the group if it doesn't exist
        if defense_type not in defense_groups:
            defense_groups[defense_type] = {
                "Non-Backdoored AUROC": None,
                "Backdoored AUROC": None,
                "Non-Backdoored F1": None,
                "Backdoored F1": None,
                "Non-Backdoored Recall@1%": None,
                "Backdoored Recall@1%": None,
            }

        # Add metrics to the appropriate group
        if "Non-Backdoored" in defense_name:
            defense_groups[defense_type]["Non-Backdoored AUROC"] = defense.get("AUROC", None)
            defense_groups[defense_type]["Non-Backdoored F1"] = defense.get("F1", None)
            defense_groups[defense_type]["Non-Backdoored Recall@1%"] = defense.get("Recall @ 1% FPR", None)
        elif "Backdoored" in defense_name:
            defense_groups[defense_type]["Backdoored AUROC"] = defense.get("AUROC", None)
            defense_groups[defense_type]["Backdoored F1"] = defense.get("F1", None)
            defense_groups[defense_type]["Backdoored Recall@1%"] = defense.get("Recall @ 1% FPR", None)

# Print the grouped results
for defense_type, metrics in defense_groups.items():
    auroc_non_backdoored = metrics["Non-Backdoored AUROC"]
    auroc_backdoored = metrics["Backdoored AUROC"]
    f1_non_backdoored = metrics["Non-Backdoored F1"]
    f1_backdoored = metrics["Backdoored F1"]
    recall_non_backdoored = metrics["Non-Backdoored Recall@1%"]
    recall_backdoored = metrics["Backdoored Recall@1%"]

    # Format metrics for display
    auroc_non_backdoored_str = f"{auroc_non_backdoored:.4f}" if auroc_non_backdoored is not None else "N/A"
    auroc_backdoored_str = f"{auroc_backdoored:.4f}" if auroc_backdoored is not None else "N/A"
    f1_non_backdoored_str = f"{f1_non_backdoored:.4f}" if f1_non_backdoored is not None else "N/A"
    f1_backdoored_str = f"{f1_backdoored:.4f}" if f1_backdoored is not None else "N/A"
    recall_non_backdoored_str = f"{recall_non_backdoored:.4f}" if recall_non_backdoored is not None else "N/A"
    recall_backdoored_str = f"{recall_backdoored:.4f}" if recall_backdoored is not None else "N/A"

    print(
        f"{defense_type:<40} | {auroc_non_backdoored_str:<15} | {auroc_backdoored_str:<15} | "
        + f"{f1_non_backdoored_str:<15} | {recall_non_backdoored_str:<15}"
    )

print("-" * 100)

# %%

# Create visualizations to compare probe performance

# Extract metrics for plotting
probe_types = []
non_backdoored_aurocs = []
backdoored_aurocs = []
non_backdoored_recalls = []
backdoored_recalls = []

for defense_type, metrics in defense_groups.items():
    if all(metrics[key] is not None for key in ["Non-Backdoored AUROC", "Backdoored AUROC"]):
        probe_types.append(defense_type)
        non_backdoored_aurocs.append(metrics["Non-Backdoored AUROC"])
        backdoored_aurocs.append(metrics["Backdoored AUROC"])

        # Add recall values if available, otherwise use 0
        non_backdoored_recalls.append(
            metrics["Non-Backdoored Recall@1%"] if metrics["Non-Backdoored Recall@1%"] is not None else 0
        )
        backdoored_recalls.append(metrics["Backdoored Recall@1%"] if metrics["Backdoored Recall@1%"] is not None else 0)

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot AUROC comparison
x = np.arange(len(probe_types))
width = 0.35

ax1.bar(x - width / 2, non_backdoored_aurocs, width, label="Non-Backdoored")
ax1.bar(x + width / 2, backdoored_aurocs, width, label="Backdoored")

ax1.set_ylabel("AUROC")
ax1.set_title("AUROC by Probe Type")
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace("Probe", "").replace("on", "") for p in probe_types], rotation=45, ha="right")
ax1.legend()
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Plot Recall@1%FPR comparison
ax2.bar(x - width / 2, non_backdoored_recalls, width, label="Non-Backdoored")
ax2.bar(x + width / 2, backdoored_recalls, width, label="Backdoored")

ax2.set_ylabel("Recall @ 1% FPR")
ax2.set_title("Recall @ 1% FPR by Probe Type")
ax2.set_xticks(x)
ax2.set_xticklabels([p.replace("Probe", "").replace("on", "") for p in probe_types], rotation=45, ha="right")
ax2.legend()
ax2.set_ylim(0, 1.0)
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# Create a scatter plot comparing AUROC vs Recall@1%FPR
plt.figure(figsize=(10, 8))

# Plot non-backdoored points
plt.scatter(
    non_backdoored_aurocs,
    non_backdoored_recalls,
    s=100,
    c="blue",
    marker="o",
    alpha=0.7,
    label="Non-Backdoored",
)

# Plot backdoored points
plt.scatter(backdoored_aurocs, backdoored_recalls, s=100, c="red", marker="x", alpha=0.7, label="Backdoored")

# Add labels for each point
for i, probe_type in enumerate(probe_types):
    plt.annotate(
        probe_type.replace("Probe", "").replace("on", ""),
        (non_backdoored_aurocs[i], non_backdoored_recalls[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    plt.annotate(
        probe_type.replace("Probe", "").replace("on", ""),
        (backdoored_aurocs[i], backdoored_recalls[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.xlabel("AUROC")
plt.ylabel("Recall @ 1% FPR")
plt.title("AUROC vs Recall @ 1% FPR for Different Probe Types")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()
