import torch
from torch import Tensor


def _area_mean(mask: Tensor) -> Tensor:
    # the denominator of the area normalization. An empty area (an all-black mask with unmasked_weight 0, or the
    # unmasked area of an all-white mask, which every image without a mask file gets) has only zero losses, and
    # 0 / 0 made the whole loss NaN. The floor (the smallest normal float) only replaces a mean of 0: a mask with any
    # value above 0 has a far larger mean, so every result that was finite before is bit-identical.
    return mask.mean(dim=(1, 2, 3), keepdim=True).clamp_min(torch.finfo(mask.dtype).tiny)


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / _area_mean(clamped_mask)

    return losses


def masked_losses_with_prior(
        losses: Tensor,
        prior_losses: Tensor | None,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        masked_prior_preservation_weight: float,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / _area_mean(clamped_mask)

    if masked_prior_preservation_weight == 0 or prior_losses is None:
        return losses

    clamped_mask = (1 - clamped_mask)
    prior_losses *= clamped_mask * masked_prior_preservation_weight

    if normalize_masked_area_loss:
        prior_losses = prior_losses / _area_mean(clamped_mask)

    return losses + prior_losses
