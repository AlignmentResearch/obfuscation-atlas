# type: ignore
"""Score aggregation and layer selection utilities."""

import torch


def aggregate_scores(layerwise_scores: dict[str, torch.Tensor], layer_aggregation: str):
    scores = torch.stack(list(layerwise_scores.values()))
    assert len(scores) > 0
    # Type checker doesn't take into account that scores is non-empty,
    # so thinks this might be a float.
    if layer_aggregation == "mean":
        return scores.mean(dim=0)
    elif layer_aggregation == "max":
        return torch.amax(scores, dim=0)
    elif layer_aggregation.startswith("mean_with_top_k"):
        try:
            k, topk_coef = layer_aggregation.split("-")[1:]
            k, topk_coef = int(k), float(topk_coef)
            k = min(k, scores.shape[0])
        except ValueError:
            raise ValueError(
                "Invalid layer aggregation format for topk. Expected format: "
                f"mean_with_top_k-<k>-<topk_coef>, got {layer_aggregation}"
            )
        topk_scores = torch.topk(scores, k=k, dim=0).values
        return topk_coef * topk_scores.mean(dim=0) + (1 - topk_coef) * scores.mean(dim=0)
    else:
        raise ValueError(f"Unknown layer aggregation: {layer_aggregation}")


def get_activation_matching_layers(activation_matching_layers: str, num_hidden_layers: int):
    if activation_matching_layers.lower() == "all":
        activation_layers = list(range(num_hidden_layers))
    elif activation_matching_layers.lower().startswith("range_"):
        range_args = map(int, activation_matching_layers.split("_")[1].split("-"))
        activation_layers = list(range(*range_args))
    elif "," in activation_matching_layers:
        activation_layers = [int(i) for i in activation_matching_layers.split(",")]
    else:
        activation_layers = [int(activation_matching_layers)]
    return activation_layers
