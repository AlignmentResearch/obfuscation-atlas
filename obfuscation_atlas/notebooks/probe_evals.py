# type: ignore
import copy
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from accelerate import find_executable_batch_size
from peft import PeftModel
from sklearn.metrics import f1_score, recall_score

from .utils import convert_float16, convert_to_serializable, get_valid_indices


def get_probe_scores(
    probes,
    encoder,
    examples,
    batch_size,
    max_length,
    device="cuda",
    probe_layers=None,
    verbose=True,
):
    # If probe_layers is not defined, set it to all the layers
    if probe_layers is None:
        probe_layers = list(probes.keys())

    # Cache activations for a set of examples using the encoder
    initial_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"  # Use right padding

    @find_executable_batch_size(starting_batch_size=batch_size)
    def get_activations(batch_size):
        return encoder.get_model_residual_acts(
            examples,
            batch_size=batch_size,
            max_length=max_length,
            return_tokens=True,
            only_return_layers=probe_layers,
            verbose=verbose,
        )

    activations, tokens = get_activations()
    encoder.tokenizer.padding_side = initial_padding_side

    return get_probe_scores_from_activations(
        probes, activations, tokens, probe_layers, batch_size, encoder.tokenizer, device
    )


def get_probe_scores_from_activations(
    probes,
    activations,
    tokens,
    probe_layers,
    batch_size,
    tokenizer,
    device="cuda",
):
    # Get probe scores for a set of examples
    probe_scores = {}

    @find_executable_batch_size(starting_batch_size=batch_size)
    def get_probe_scores_batch_size(batch_size):
        for layer in probe_layers:
            probe = probes[layer]
            probe.to(device)
            probe.eval()  # Set the probe to evaluation mode

            layer_activations = activations[layer]
            n_examples = len(layer_activations)
            layer_scores = []

            with torch.no_grad():  # Disable gradient computation for inference
                for i in range(0, n_examples, batch_size):
                    batch = layer_activations[i : i + batch_size].to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        batch_scores = probe.predict(batch)
                        batch_scores = (batch_scores.detach().cpu().numpy() * 2 - 1) * 3
                    layer_scores.append(batch_scores)

            probe_scores[layer] = np.concatenate(layer_scores)
            probe.to("cpu")  # Move the probe back to CPU to free up GPU memory
        return probe_scores

    probe_scores = get_probe_scores_batch_size()
    activations.clear()

    # Get the (token, score) pairs for each example
    paired_scores = {}
    for layer, scores in probe_scores.items():
        paired_scores[layer] = [
            [
                (
                    tokenizer.decode(tokens["input_ids"][example_idx][token_idx].item()),
                    scores[example_idx][token_idx],
                )
                for token_idx in range(tokens["input_ids"].shape[1])
                if tokens["attention_mask"][example_idx][token_idx].item()  # Skip padding tokens
            ]
            for example_idx in range(tokens["input_ids"].shape[0])
        ]

    return paired_scores


def remove_scores_between_tokens(paired_scores_all_splits, only_return_on_tokens_between, tokenizer=None):
    paired_scores_all_splits_copy = copy.deepcopy(paired_scores_all_splits)

    for paired_scores in paired_scores_all_splits_copy.values():
        first_layer = next(iter(paired_scores))

        for example_idx, example_data in enumerate(paired_scores[first_layer]):
            if tokenizer is not None:
                tokens = [tokenizer.encode(token, add_special_tokens=False)[0] for token, _ in example_data]
            else:
                tokens = [token for token, _ in example_data]

            valid_indices = set(get_valid_indices(tokens, only_return_on_tokens_between))

            for layer_data in paired_scores.values():
                layer_data[example_idx] = [
                    [token, score if i in valid_indices else None]
                    for i, (token, score) in enumerate(layer_data[example_idx])
                ]

    return paired_scores_all_splits_copy


def get_annotated_dataset(
    probes,
    encoder,
    dataset,
    splits,
    batch_size,
    max_length,
    model_adapter_path=None,
    **kwargs,
):
    # Load model adapter if provided
    if model_adapter_path is not None:
        print("Loading model adapter...")
        assert not isinstance(encoder.model, PeftModel)  # model should not be a PeftModel at this point
        encoder.model = PeftModel.from_pretrained(encoder.model, model_adapter_path)
        # encoder.model = encoder.model.merge_and_unload()

    # Get scores
    scores_dict = {}
    dataset_splits = {split: dataset[split].select(range(min(3000, len(dataset[split])))) for split in splits}
    for split in splits:
        print(split)

        split_dataset = dataset_splits[split]
        split_dataset_str = [
            split_dataset[i]["prompt"] + split_dataset[i]["completion"] for i in range(len(split_dataset))
        ]

        with torch.no_grad():
            paired_scores = get_probe_scores(
                probes=probes,
                encoder=encoder,
                examples=split_dataset_str,
                batch_size=batch_size,
                max_length=max_length,
                **kwargs,
            )
        scores_dict[split] = paired_scores

    if model_adapter_path is not None:
        # remove the lora adapter
        encoder.model = encoder.model.base_model

    return convert_float16(scores_dict)


def load_or_create_annotated_dataset(
    probes_folder,
    file_name,
    probes,
    encoder,
    jailbreaks_dataset,
    datasets,
    model_adapter_path=None,
    max_length=8192,
    batch_size=16,
):
    annotations_folder_name = os.path.join(probes_folder, file_name)

    if os.path.exists(annotations_folder_name):
        with open(annotations_folder_name, "r") as f:
            return json.load(f)
    else:
        kwargs = {"max_length": max_length, "batch_size": batch_size}

        if model_adapter_path:
            kwargs["model_adapter_path"] = model_adapter_path

        scores_dict = get_annotated_dataset(probes, encoder, jailbreaks_dataset, datasets, **kwargs)

        # Save with enhanced serialization
        with open(annotations_folder_name, "w") as f:
            json.dump(scores_dict, f, default=convert_to_serializable)

        return scores_dict


def load_annotated_dataset(
    probes_folder,
    file_name,
):
    annotations_folder_name = os.path.join(probes_folder, file_name)

    if os.path.exists(annotations_folder_name):
        with open(annotations_folder_name, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"File {annotations_folder_name} does not exist.")


def vickrey_auc(scores, k):
    # Compute the Vickrey AUC for a list of scores
    # Returns the k-th highest score
    if not scores:
        raise ValueError("Scores list must be non-empty")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    k = min(k, len(scores))
    return sorted(scores, reverse=True)[k - 1]


def aggregate_across_layers(all_split_scores, layers, cross_layer_aggregation):
    # Given the probe scores foor multiple layers, compute a single score for each token
    aggregation_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "vickrey": lambda x: vickrey_auc(x, 2),
    }

    if cross_layer_aggregation not in aggregation_funcs:
        raise ValueError(f"Invalid cross_layer_aggregation: {cross_layer_aggregation}")

    aggregation_func = aggregation_funcs[cross_layer_aggregation]

    def wrapped_aggregation_func(layer_scores):
        if None in layer_scores:
            return None
        return aggregation_func(layer_scores)

    new_all_split_scores = {}
    for split, split_scores in all_split_scores.items():
        split_scores = {str(k): v for k, v in split_scores.items()}
        new_split_scores = []

        for example in zip(*(split_scores[str(layer)] for layer in layers)):
            new_example = [
                (
                    token_scores[0][0],
                    wrapped_aggregation_func([score for _, score in token_scores]),
                )
                for token_scores in zip(*example)
            ]
            new_split_scores.append(new_example)

        new_all_split_scores[split] = new_split_scores

    return new_all_split_scores


def aggregate_across_tokens(all_split_scores, cross_token_aggregation):
    # Given the probe scores for each token, compute a single score for each example
    aggregation_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "vickrey": lambda x: vickrey_auc(x, 2),
        "logsumexp": lambda x: np.log(np.exp(x).sum()),
        "top_2_percent": lambda x: vickrey_auc(x, int(0.02 * len(x)) + 1),
        "median_over_zero": lambda x: np.median([score for score in x if score > 0]),
    }

    if cross_token_aggregation not in aggregation_funcs:
        raise ValueError(f"Invalid cross_token_aggregation: {cross_token_aggregation}")

    aggregation_func = aggregation_funcs[cross_token_aggregation]

    aggregated_scores = {}
    min_score, max_score = float("inf"), float("-inf")

    for split, split_scores in all_split_scores.items():
        if isinstance(split_scores, dict):
            aggregated_scores[split] = {}
            for layer in split_scores:
                new_split_scores = []
                for example in split_scores[layer]:
                    example_scores = [score for _, score in example if score is not None]
                    example_scalar = aggregation_func(example_scores)
                    new_split_scores.append(example_scalar)

                    min_score = min(min_score, example_scalar)
                    max_score = max(max_score, example_scalar)

                aggregated_scores[split][layer] = new_split_scores

        elif isinstance(split_scores, list):
            new_split_scores = []
            for example in split_scores:
                example_scores = [score for _, score in example if score is not None]
                example_scalar = aggregation_func(example_scores)
                new_split_scores.append(example_scalar)

                min_score = min(min_score, example_scalar)
                max_score = max(max_score, example_scalar)

            aggregated_scores[split] = new_split_scores

    return aggregated_scores, min_score, max_score


def compute_recall(threshold, aggregated_scores, negative_splits, positive_splits):
    # Compute the recall for a given threshold
    y_true, y_pred = [], []
    for split in negative_splits + positive_splits:
        scores = aggregated_scores[split]
        y_true.extend([0 if split in negative_splits else 1] * len(scores))
        y_pred.extend([int(score >= threshold) for score in scores])
    return recall_score(y_true, y_pred)


def compute_f1(threshold, aggregated_scores, negative_splits, positive_splits):
    # Compute the F1 score for a given threshold
    y_true, y_pred = [], []
    for split in negative_splits + positive_splits:
        scores = aggregated_scores[split]
        y_true.extend([0 if split in negative_splits else 1] * len(scores))
        y_pred.extend([int(score >= threshold) for score in scores])
    return f1_score(y_true, y_pred)


def find_best_threshold(aggregated_scores, negative_splits, positive_splits, min_score, max_score):
    # Find the best threshold for the aggregated scores
    thresholds = np.linspace(min_score, max_score, 1000)
    f1_scores = [compute_f1(t, aggregated_scores, negative_splits, positive_splits) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def get_threshold_at_fpr(aggregated_scores, negative_splits, min_score, max_score, target_fpr):
    # Compute the minimum threshold that achieves a given false positive rate
    thresholds = np.linspace(min_score, max_score, 1000)

    final_threshold = max_score
    for threshold in reversed(thresholds):
        # Calculate false positive rate at this threshold
        fp = sum(score >= threshold for split in negative_splits for score in aggregated_scores[split])
        tn = sum(score < threshold for split in negative_splits for score in aggregated_scores[split])
        fpr = fp / (fp + tn)

        # If we've reached or exceeded the target FPR, return this threshold
        if fpr <= target_fpr:
            final_threshold = threshold

    return final_threshold


def create_scores_plot(
    aggregated_scores,
    best_threshold,
    best_f1,
    title,
    false_positive_rate,
    allowed_labels,
):
    plt.figure(figsize=(12, 6))

    data = [aggregated_scores[label] for label in allowed_labels]
    labels = allowed_labels
    colors = sns.color_palette("husl", n_colors=len(aggregated_scores))

    # Create violin plot
    parts = plt.violinplot(data, vert=False, showmeans=False, showextrema=False, showmedians=False)

    # Customize violin plot colors and add quartile lines
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

        # Calculate quartiles
        quartile1, median, quartile3 = np.percentile(data[i], [25, 50, 75])

        # Add quartile lines
        plt.hlines(i + 1, quartile1, quartile3, color="k", linestyle="-", lw=5, alpha=0.7)
        plt.vlines(median, i + 0.95, i + 1.05, color="white", linestyle="-", lw=2)

    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best Threshold: {best_threshold:.2f}",
    )

    plt.title(f"{title}\nOverall F1: {best_f1:.2f}", fontsize=14)
    plt.xlabel("Aggregated Score", fontsize=12)
    plt.yticks(range(1, len(labels) + 1), labels, fontsize=10)

    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold at {false_positive_rate * 100:.2f}% FPR",
        ),
        plt.Line2D([0], [0], color="k", linestyle="-", lw=5, alpha=0.7, label="IQR"),
        plt.Line2D([0], [0], color="white", linestyle="-", lw=2, label="Median"),
    ]
    legend_elements.extend(
        [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=label) for label, color in zip(labels, colors)]
    )

    plt.legend(
        handles=legend_elements,
        fontsize=8,
        title="Categories",
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()
    plt.show()


def get_per_split_scores(
    aggregated_scores,
    best_threshold,
    positive_splits,
    negative_splits,
    train_positive_splits,
    train_negative_splits,
):
    per_split_scores = {}

    for split in positive_splits + negative_splits + train_positive_splits + train_negative_splits:
        if split not in aggregated_scores:
            print(f"Warning: Split '{split}' not found in aggregated_scores.")
            continue

        scores = aggregated_scores[split]
        total_samples = len(scores)

        if split in positive_splits:
            correct_predictions = sum(score >= best_threshold for score in scores)
        else:  # negative_splits and heldout_splits
            correct_predictions = sum(score < best_threshold for score in scores)

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        per_split_scores[split] = accuracy

    return per_split_scores


def generate_score_plots(
    all_split_scores,
    positive_splits,
    negative_splits,
    train_positive_splits,
    train_negative_splits,
    layers,
    cross_token_aggregation,
    cross_layer_aggregation=None,
    false_positive_rate=0.05,
    title="",
    show=True,
):
    """
    Modified from the original code to use return the threshold and recall
    at the false positive rate while still also returning the best f1 score.
    """
    if cross_layer_aggregation:
        all_split_scores = aggregate_across_layers(all_split_scores, layers, cross_layer_aggregation)

    aggregated_scores, min_score, max_score = aggregate_across_tokens(all_split_scores, cross_token_aggregation)

    best_f1_threshold, _ = find_best_threshold(
        aggregated_scores, train_negative_splits, train_positive_splits, min_score, max_score
    )
    best_f1 = compute_f1(best_f1_threshold, aggregated_scores, negative_splits, positive_splits)

    fpr_threshold = get_threshold_at_fpr(
        aggregated_scores, train_negative_splits, min_score, max_score, false_positive_rate
    )
    recall_at_fpr = compute_recall(fpr_threshold, aggregated_scores, negative_splits, positive_splits)

    if show:
        create_scores_plot(
            aggregated_scores,
            fpr_threshold,
            best_f1,
            title,
            false_positive_rate,
            positive_splits + negative_splits + train_positive_splits + train_negative_splits,
        )

    per_split_scores = get_per_split_scores(
        aggregated_scores,
        fpr_threshold,
        positive_splits,
        negative_splits,
        train_positive_splits,
        train_negative_splits,
    )

    return (
        list(aggregated_scores.values()),
        list(aggregated_scores.keys()),
        fpr_threshold,
        best_f1,
        per_split_scores,
        recall_at_fpr,
    )
