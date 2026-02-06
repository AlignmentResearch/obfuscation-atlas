import warnings
from collections import defaultdict
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from matplotlib.figure import Figure
from scipy import stats

from obfuscation_atlas.utils.example_types import ExampleType


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, false_positive_rate: float) -> float:
    """Calculate the recall (true positive rate) at a specified false positive rate.
    Gives a conservative estimate (highest recall where FPR ≤ target FPR).

    Args:
        scores: Array of prediction scores (higher scores indicate positive class)
        labels: Array of true binary labels (0 or 1)
        false_positive_rate: Target false positive rate (between 0 and 1)

    Returns:
        The recall (true positive rate) at the specified false positive rate

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not 0 <= false_positive_rate <= 1:
        raise ValueError("false_positive_rate must be between 0 and 1")

    # Get FPR, TPR, and thresholds from ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)

    # Find the highest TPR where FPR <= target false_positive_rate
    valid_indices = np.where(fpr <= false_positive_rate)[0]

    if len(valid_indices) == 0:
        # If no threshold achieves the target FPR, return 0
        return 0.0

    # Get the highest recall at the target FPR
    recall = tpr[valid_indices[-1]]

    return float(recall)


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Mapping[str, float]:
    """Calculate metrics at a given threshold."""
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    precision = sklearn.metrics.precision_score(labels, predictions)
    recall = sklearn.metrics.recall_score(labels, predictions)
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.0
    return {
        "threshold": float(threshold),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def best_threshold_for_metric(
    scores: np.ndarray, labels: np.ndarray, metric_name: str, exclude_scores: bool = False
) -> dict[str, Any]:
    """Find the best classification threshold for a given metric and return all associated metrics.

    Args:
        scores: An array of prediction scores from a classifier.
        labels: An array of true binary labels (0 or 1).
        metric_name: The name of the metric to optimize for. Must be one of
                     "f1_score", "precision", "recall", or "accuracy".

    Returns:
        A dictionary containing the optimal threshold and the f1_score, precision,
        recall, and accuracy at that threshold.
    """
    valid_metrics = ["f1_score", "precision", "recall", "accuracy"]
    if metric_name not in valid_metrics:
        raise ValueError(f"metric_name must be one of {valid_metrics}, but got '{metric_name}'")

    # Use sklearn's precision_recall_curve to get thresholds and corresponding metrics
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores)

    # To align with the thresholds array, we discard the last metric value
    # which corresponds to a threshold of 1.0 and recall of 0.0.
    precision = precision[:-1]
    recall = recall[:-1]

    # Calculate F1 scores, ignoring division-by-zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    # Calculate accuracy for each threshold
    accuracies = np.array([np.mean((scores >= t).astype(int) == labels) for t in thresholds])

    # Map the metric name to its corresponding calculated array
    metric_map = {
        "f1_score": f1_scores,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracies,
    }

    # Find the index of the threshold that maximizes the chosen metric
    best_idx = np.argmax(metric_map[metric_name])

    # Retrieve all metrics at that optimal index
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_accuracy = accuracies[best_idx]

    ret: dict[str, Any] = {
        "threshold": float(best_threshold),
        "f1_score": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "accuracy": float(best_accuracy),
    }
    if not exclude_scores:
        ret["scores"] = {
            "positive": scores[labels == 1].tolist(),
            "negative": scores[labels == 0].tolist(),
        }
    return ret


def get_detector_metrics(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    false_positive_rate: float = 0.01,
    thresholds: Mapping[str, float] | None = None,
    disable_plots: bool = True,
    histogram_percentile: float = 95,
    num_bins: int = 100,
    log_yaxis: bool = True,
    flat_dict: bool = False,
    exclude_scores: bool = False,
    example_types: np.ndarray | None = None,
    logits: Mapping[str, np.ndarray] | None = None,
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Figure]]:
    """Calculate detector metrics for each layer.

    Args:
        scores: Mapping of layer name to score array
        labels: Binary labels (0 or 1)
        false_positive_rate: Target FPR for recall calculation
        thresholds: Optional thresholds per layer
        disable_plots: Whether to skip generating plots
        histogram_percentile: Percentile for histogram upper limit
        num_bins: Number of histogram bins
        log_yaxis: Whether to use log scale on y-axis
        flat_dict: Whether to flatten metrics dict
        exclude_scores: Whether to exclude raw scores from output
        example_types: Optional array of ExampleType values (0=CONGRUENT_POSITIVE,
            1=CONGRUENT_NEGATIVE, 2=INCONGRUENT_POSITIVE, 3=INCONGRUENT_NEGATIVE)
            for breaking down scores by congruent vs incongruent
        logits: Optional mapping of layer name to logit array for computing
            logit-space metrics (e.g., Wasserstein distance in logit space)

    Returns:
        Tuple of (metrics dict, figures dict)
    """
    metrics = defaultdict(dict)
    figs = {}

    for layer in scores:
        auc_roc = sklearn.metrics.roc_auc_score(
            y_true=labels,
            y_score=scores[layer],
        )
        ap = sklearn.metrics.average_precision_score(
            y_true=labels,
            y_score=scores[layer],
        )
        metrics[layer]["AUC_ROC"] = auc_roc
        metrics[layer]["AP"] = ap
        metrics[layer][f"recall_at_{int(false_positive_rate * 100)}%_fpr"] = recall_at_fpr(
            scores[layer], labels, false_positive_rate
        )
        # Compute score arrays once for reuse
        pos_scores = scores[layer][labels == 1]
        neg_scores = scores[layer][labels == 0]

        metrics[layer]["mean_score_positive"] = float(pos_scores.mean())
        metrics[layer]["mean_score_negative"] = float(neg_scores.mean())
        metrics[layer]["score_separation"] = (
            metrics[layer]["mean_score_positive"] - metrics[layer]["mean_score_negative"]
        )
        # Wasserstein distance (Earth Mover's Distance) - measures work to transform one distribution to another
        metrics[layer]["wasserstein_distance"] = float(stats.wasserstein_distance(pos_scores, neg_scores))

        # Wasserstein distance in logit space - more sensitive to differences near probability 0 and 1
        if logits is not None and layer in logits:
            pos_logits = logits[layer][labels == 1]
            neg_logits = logits[layer][labels == 0]
            metrics[layer]["score_separation_logit"] = float(pos_logits.mean() - neg_logits.mean())
            metrics[layer]["wasserstein_distance_logit"] = float(stats.wasserstein_distance(pos_logits, neg_logits))

        # Kolmogorov-Smirnov statistic - max CDF difference between distributions
        ks_stat, ks_pvalue = stats.ks_2samp(pos_scores, neg_scores)
        metrics[layer]["ks_statistic"] = float(ks_stat)
        metrics[layer]["ks_pvalue"] = float(ks_pvalue)

        f1_metrics = (
            best_threshold_for_metric(scores[layer], labels, "f1_score", exclude_scores=True)
            if thresholds is None
            else metrics_at_threshold(scores[layer], labels, thresholds[layer])
        )
        if flat_dict:
            metrics[layer].update(f1_metrics)
        else:
            metrics[layer]["metrics_at_threshold"] = f1_metrics

        # Save the scores for the positive and negative examples in metrics
        if not exclude_scores:
            metrics[layer]["scores"] = {
                "positive": scores[layer][labels == 1].tolist(),
                "negative": scores[layer][labels == 0].tolist(),
            }
            # Add breakdown by example type if provided
            if example_types is not None:
                metrics[layer]["scores_by_type"] = {
                    et.key: scores[layer][example_types == et].tolist() for et in ExampleType
                }
        if not disable_plots:
            upper_lim = np.percentile(scores[layer], histogram_percentile).item()
            # Usually there aren't extremely low outliers, so we just use the minimum,
            # otherwise this tends to weirdly cut of the histogram.
            lower_lim = scores[layer].min().item()

            bins = np.linspace(lower_lim, upper_lim, num_bins)

            # Visualizations for anomaly scores
            fig, ax = plt.subplots()
            for i, name in enumerate(["Normal", "Anomalous"]):
                vals = scores[layer][labels == i]
                ax.hist(
                    vals,
                    bins=bins,
                    alpha=0.5,
                    label=name,
                    log=log_yaxis,
                )
            ax.legend()
            ax.set_xlabel("Anomaly score")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Anomaly score distribution ({layer})")
            textstr = f"AUROC: {auc_roc:.1%}\n AP: {ap:.1%}"
            props = dict(boxstyle="round", facecolor="white")
            ax.text(
                0.98,
                0.80,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=props,
            )
            figs[layer] = fig
    return metrics, figs
