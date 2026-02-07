from collections.abc import Callable
from dataclasses import dataclass

import torch

from afterburner.grpo_config import KLEstimator, LossAggregation

# Loss function registry
LOSS_FUNCTIONS: dict[str, Callable] = {}


def register_loss(name: str):
    """Decorator to register a loss function."""

    def decorator(func):
        LOSS_FUNCTIONS[name] = func
        return func

    return decorator


def get_kl_estimator(estimator: KLEstimator) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the KL estimator function to apply to log pi - log pi_ref."""
    match estimator:
        case KLEstimator.SQUARED:
            return lambda x: 0.5 * x**2
        case KLEstimator.VAR_REDUCED:
            return lambda x: x + torch.exp(-x) - 1
        case KLEstimator.VANILLA:
            return lambda x: x
        case _:
            raise ValueError(f"Unknown KL estimator: {estimator}")


@dataclass(frozen=True)
class Loss:
    """Loss object containing the loss for backprop, sample metrics for table logging and group metrics for wandb plots."""

    loss: torch.Tensor
    group_metrics: dict[str, torch.Tensor]

    def __post_init__(self):
        assert self.loss.requires_grad, "Loss should be trainable"
        assert all(not metric.requires_grad for metric in self.group_metrics.values()), "All group metrics should be detached"


def masked_batch_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of a tensor, masking out the padded tokens."""
    return (x * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def get_loss_aggregator(
    aggregation: LossAggregation, mask: torch.Tensor, prompt_indices: torch.Tensor, max_response_length: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the loss aggregator function to aggregate the loss over the batch.

    Args:
        aggregation: The aggregation method to use. Taxonomy is taken from https://arxiv.org/pdf/2510.13786#page=7.
        mask: The completion mask tensor of shape [batch_size, seq_len].
        prompt_indices: The prompt indices tensor of shape [batch_size].
        max_response_length: The maximum response length (used for Doctor aggregation).

    Returns:
        A function that takes a metric tensor of shape [batch_size, seq_len] and returns a scalar loss.
    """
    match aggregation:
        case LossAggregation.SAMPLE:
            return lambda x: masked_batch_mean(x, mask).mean()
        case LossAggregation.PROMPT:

            def aggregate_by_prompt(x: torch.Tensor) -> torch.Tensor:
                unique_prompts = torch.unique(prompt_indices)
                prompt_losses = []
                for prompt_idx in unique_prompts:
                    prompt_mask = prompt_indices == prompt_idx
                    prompt_losses.append((x * mask)[prompt_mask].sum() / mask[prompt_mask].sum().clamp(min=1.0))
                return torch.stack(prompt_losses).mean()

            return aggregate_by_prompt
        case LossAggregation.TOKEN:
            return lambda x: (x * mask).sum() / mask.sum().clamp(min=1.0)
        case LossAggregation.DOCTOR:
            return lambda x: ((x * mask).sum(-1) / max_response_length).mean()


@register_loss("grpo")
def compute_grpo_loss(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute GRPO loss with advantages and clipped objectives."""
    loss_aggregator = get_loss_aggregator(loss_aggregation, completion_mask, prompt_indices, max_response_length)

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(log_ratio)  # [batch_size, seq_len-1]

    # GRPO clipped objective with proper epsilon values
    coef1 = ratio
    coef2 = torch.clamp(coef1, 1.0 - epsilon, 1.0 + epsilon_high)

    # Two-sided clipping
    if delta is not None:
        coef1 = torch.clamp(coef1, max=delta)

    per_token_loss1 = coef1 * advantages.unsqueeze(1)
    per_token_loss2 = coef2 * advantages.unsqueeze(1)
    per_token_advantage_loss = -torch.min(per_token_loss1, per_token_loss2)  # [batch_size, seq_len-1]

    # KL penalty
    kl_function = get_kl_estimator(kl_estimator)
    per_token_kl = kl_function(policy_logprobs - reference_logprobs)
    per_token_kl_loss = per_token_kl * kl_coeff

    # Other metrics
    is_low_clipped = (coef1 < 1 - epsilon) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    return Loss(
        loss=loss_aggregator(per_token_advantage_loss + per_token_kl_loss),
        group_metrics=dict(
            kl_loss=loss_aggregator(per_token_kl_loss).detach(),
            kl_div=loss_aggregator(per_token_kl).detach(),
            clip_low_mean=loss_aggregator(is_low_clipped.float()).detach(),
            clip_high_mean=loss_aggregator(is_high_clipped.float()).detach(),
            clip_region_mean=loss_aggregator(is_region_clipped.float()).detach(),
            mean_abs_log_ratio=loss_aggregator(torch.abs(log_ratio)).detach(),
        ),
    )


@register_loss("grpo_kl_adv")
def compute_grpo_loss_with_kl_in_advantages(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute GRPO loss with the KL added to the advantages instead of a separate loss."""
    loss_aggregator = get_loss_aggregator(loss_aggregation, completion_mask, prompt_indices, max_response_length)

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(log_ratio)  # [batch_size, seq_len-1]

    # GRPO clipped objective with proper epsilon values
    coef1 = ratio
    coef2 = torch.clamp(coef1, 1.0 - epsilon, 1.0 + epsilon_high)

    # Two-sided clipping
    if delta is not None:
        coef1 = torch.clamp(coef1, max=delta)

    # KL penalty (summed over the sequence length)
    kl_function = get_kl_estimator(kl_estimator)
    # Compute KL with detached grads for correct score function gradient estimate
    kl_divs = masked_batch_mean(kl_function(policy_logprobs.detach() - reference_logprobs), completion_mask)
    kl_losses = kl_divs * kl_coeff
    advantages -= kl_losses

    per_token_loss1 = coef1 * advantages.unsqueeze(1)
    per_token_loss2 = coef2 * advantages.unsqueeze(1)
    per_token_advantage_loss = -torch.min(per_token_loss1, per_token_loss2)  # [batch_size, seq_len-1]

    # Other metrics
    is_low_clipped = (coef1 < 1 - epsilon) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    return Loss(
        loss=loss_aggregator(per_token_advantage_loss),
        group_metrics=dict(
            kl_loss=kl_losses.mean().detach(),
            kl_div=kl_divs.mean().detach(),
            importance_sampled_kl=loss_aggregator((policy_logprobs - reference_logprobs) * ratio).detach(),
            clip_low_mean=loss_aggregator(is_low_clipped.float()).detach(),
            clip_high_mean=loss_aggregator(is_high_clipped.float()).detach(),
            clip_region_mean=loss_aggregator(is_region_clipped.float()).detach(),
            mean_abs_log_ratio=loss_aggregator(torch.abs(log_ratio)).detach(),
        ),
    )


@register_loss("gspo")
def compute_gspo_loss(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute GSPO loss with advantages and clipped objectives."""
    assert loss_aggregation == LossAggregation.SAMPLE, "GSPO only supports sample aggregation"

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(masked_batch_mean(log_ratio, completion_mask))  # [batch_size]

    # GRPO clipped objective with proper epsilon values
    coef1 = ratio
    coef2 = torch.clamp(coef1, 1.0 - epsilon, 1.0 + epsilon_high)  # [batch_size]

    # Two-sided clipping
    if delta is not None:
        coef1 = torch.clamp(coef1, max=delta)

    loss1 = coef1 * advantages
    loss2 = coef2 * advantages
    advantage_loss = -torch.min(loss1, loss2)  # [batch_size]

    # KL penalty
    kl_function = get_kl_estimator(kl_estimator)
    per_token_kl = kl_function(policy_logprobs - reference_logprobs)
    kl_div = masked_batch_mean(per_token_kl, completion_mask)
    kl_loss = kl_div * kl_coeff

    # Other metrics
    low_clip = (coef1 < 1 - epsilon) & (advantages < 0)
    high_clip = (coef1 > 1 + epsilon_high) & (advantages > 0)
    clip_ratio = low_clip | high_clip
    mean_abs_log_ratio = masked_batch_mean(torch.abs(log_ratio), completion_mask)

    return Loss(
        loss=(advantage_loss + kl_loss).mean(),
        group_metrics=dict(
            kl_loss=kl_loss.mean().detach(),
            kl_div=kl_div.mean().detach(),
            clip_low_mean=low_clip.float().mean().detach(),
            clip_high_mean=high_clip.float().mean().detach(),
            clip_region_mean=clip_ratio.float().mean().detach(),
            abs_log_ratio=mean_abs_log_ratio.mean().detach(),
        ),
    )


@register_loss("gspo_kl_adv")
def compute_gspo_loss_with_kl_in_advantages(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute GSPO loss with the KL added to the advantages instead of a separate loss."""
    assert loss_aggregation == LossAggregation.SAMPLE, "GSPO only supports sample aggregation"

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(masked_batch_mean(log_ratio, completion_mask))  # [batch_size]

    # GRPO clipped objective with proper epsilon values
    coef1 = ratio
    coef2 = torch.clamp(coef1, 1.0 - epsilon, 1.0 + epsilon_high)  # [batch_size]

    # Two-sided clipping
    if delta is not None:
        coef1 = torch.clamp(coef1, max=delta)

    # KL penalty (summed over the sequence length)
    kl_function = get_kl_estimator(kl_estimator)
    # Compute KL with detached grads for correct gradient estimate
    kl_div = masked_batch_mean(kl_function(policy_logprobs.detach() - reference_logprobs), completion_mask)
    kl_loss = kl_div * kl_coeff
    advantages -= kl_loss

    loss1 = coef1 * advantages
    loss2 = coef2 * advantages
    advantage_loss = -torch.min(loss1, loss2)  # [batch_size]

    # Total loss
    total_losses = advantage_loss

    # Other metrics
    low_clip = (coef1 < 1 - epsilon) & (advantages < 0)
    high_clip = (coef1 > 1 + epsilon_high) & (advantages > 0)
    clip_ratio = low_clip | high_clip

    mean_abs_log_ratio = masked_batch_mean(torch.abs(log_ratio), completion_mask)

    return Loss(
        loss=total_losses.mean(),  # N.B. Do not detach this one!
        group_metrics=dict(
            kl_loss=kl_loss.mean().detach(),
            kl_div=kl_div.mean().detach(),
            importance_sampled_kl=(kl_div * ratio).mean().detach(),
            clip_low_mean=low_clip.float().mean().detach(),
            clip_high_mean=high_clip.float().mean().detach(),
            clip_region_mean=clip_ratio.float().mean().detach(),
            abs_log_ratio=mean_abs_log_ratio.mean().detach(),
        ),
    )


@register_loss("cispo")
def compute_cispo_loss(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute CISPO loss with advantages and clipped objectives."""
    assert delta is None, "Delta is not supported for CISPO"
    loss_aggregator = get_loss_aggregator(loss_aggregation, completion_mask, prompt_indices, max_response_length)

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(log_ratio)  # [batch_size, seq_len-1]
    ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon_high)

    # Equation 4 from https://arxiv.org/pdf/2506.13585
    per_token_advantage_loss = -ratio.detach() * advantages.unsqueeze(1) * policy_logprobs

    # KL penalty
    kl_function = get_kl_estimator(kl_estimator)
    per_token_kl = kl_function(policy_logprobs - reference_logprobs)
    per_token_kl_loss = per_token_kl * kl_coeff

    # Other metrics
    is_low_clipped = torch.isclose(ratio, torch.tensor(1 - epsilon, device=ratio.device))
    is_high_clipped = torch.isclose(ratio, torch.tensor(1 + epsilon_high, device=ratio.device))
    is_region_clipped = is_low_clipped | is_high_clipped

    return Loss(
        loss=loss_aggregator(per_token_advantage_loss + per_token_kl_loss),
        group_metrics=dict(
            kl_loss=loss_aggregator(per_token_kl_loss).detach(),
            kl_div=loss_aggregator(per_token_kl).detach(),
            clip_low_mean=loss_aggregator(is_low_clipped.float()).detach(),
            clip_high_mean=loss_aggregator(is_high_clipped.float()).detach(),
            clip_region_mean=loss_aggregator(is_region_clipped.float()).detach(),
            mean_abs_log_ratio=loss_aggregator(torch.abs(log_ratio)).detach(),
        ),
    )


@register_loss("cispo_kl")
def compute_cispo_kl_loss(
    advantages: torch.Tensor,  # [batch_size]
    policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    reference_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    old_policy_logprobs: torch.Tensor,  # [batch_size, seq_len-1]
    completion_mask: torch.Tensor,  # [batch_size, seq_len-1]
    prompt_indices: torch.Tensor,  # [batch_size]
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: LossAggregation,
) -> Loss:
    """Compute CISPO loss with modified KL loss."""
    assert delta is None, "Delta is not supported for CISPO"
    loss_aggregator = get_loss_aggregator(loss_aggregation, completion_mask, prompt_indices, max_response_length)

    log_ratio = policy_logprobs - old_policy_logprobs
    ratio = torch.exp(log_ratio)  # [batch_size, seq_len-1]
    ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon_high)

    # Equation 4 from https://arxiv.org/pdf/2506.13585
    per_token_advantage_loss = -ratio.detach() * advantages.unsqueeze(1) * policy_logprobs

    # KL penalty
    kl_function = get_kl_estimator(kl_estimator)
    per_token_kl = kl_function(policy_logprobs - reference_logprobs)
    per_token_kl_loss = ratio.detach() * per_token_kl * kl_coeff

    # Other metrics
    is_low_clipped = torch.isclose(ratio, torch.tensor(1 - epsilon, device=ratio.device))
    is_high_clipped = torch.isclose(ratio, torch.tensor(1 + epsilon_high, device=ratio.device))
    is_region_clipped = is_low_clipped | is_high_clipped

    return Loss(
        loss=loss_aggregator(per_token_advantage_loss + per_token_kl_loss),
        group_metrics=dict(
            kl_loss=loss_aggregator(per_token_kl_loss).detach(),
            kl_div=loss_aggregator(per_token_kl).detach(),
            clip_low_mean=loss_aggregator(is_low_clipped.float()).detach(),
            clip_high_mean=loss_aggregator(is_high_clipped.float()).detach(),
            clip_region_mean=loss_aggregator(is_region_clipped.float()).detach(),
            mean_abs_log_ratio=loss_aggregator(torch.abs(log_ratio)).detach(),
        ),
    )
