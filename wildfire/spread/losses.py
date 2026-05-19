"""Loss functions tuned for fire-pixel segmentation imbalance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FireSpreadLoss(nn.Module):
    """Weighted BCE + soft Dice — standard combo for sparse fire masks."""

    def __init__(self, fire_pos_weight: float = 5.0, dice_weight: float = 0.5):
        super().__init__()
        self.fire_pos_weight = fire_pos_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor([self.fire_pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        # Soft Dice over the fire channel
        intersection = (probs * target).sum(dim=(1, 2, 3))
        cardinality  = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

        return bce + self.dice_weight * dice
