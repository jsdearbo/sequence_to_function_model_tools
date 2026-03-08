"""
Custom loss functions for genomic sequence-to-function models.

Includes losses designed for splicing prediction (PSI-aware),
distributional comparison (Bhattacharyya), and masked multitask training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSILoss(nn.Module):
    """
    PSI-aware loss for splicing quantification models.

    Penalizes prediction errors weighted by proximity to the decision
    boundary (PSI ≈ 0.5), where accurate discrimination matters most.
    Optionally masks bins without splicing event coverage.

    Args:
        lambda_mid: Weight multiplier for events near PSI = 0.5.
        eps: Small constant to avoid log(0).
        use_mask: If True, only compute loss on bins with event coverage.
    """

    def __init__(
        self,
        lambda_mid: float = 25.0,
        eps: float = 1e-6,
        use_mask: bool = True,
    ):
        super().__init__()
        self.lambda_mid = lambda_mid
        self.eps = eps
        self.use_mask = use_mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted PSI values, shape (B, T, L) or (B, L).
            target: Ground truth PSI values, same shape as pred.
            mask: Binary mask (1 = valid bin), same shape as pred.
        """
        # Clamp predictions to valid probability range
        pred = pred.clamp(self.eps, 1.0 - self.eps)

        # Base BCE loss
        bce = -(
            target * torch.log(pred + self.eps)
            + (1.0 - target) * torch.log(1.0 - pred + self.eps)
        )

        # Weight by proximity to decision boundary
        # Events near PSI=0.5 are hardest and most informative
        midpoint_weight = 1.0 + self.lambda_mid * torch.exp(
            -4.0 * (target - 0.5) ** 2
        )
        loss = bce * midpoint_weight

        if self.use_mask and mask is not None:
            loss = loss * mask
            n_valid = mask.sum().clamp(min=1)
            return loss.sum() / n_valid

        return loss.mean()


class BhattacharyyaLoss(nn.Module):
    """
    Bin-level Bhattacharyya distance loss for multitask genomic prediction.

    At each output bin, normalizes predictions and targets across the
    task dimension to form distributions, then computes the Bhattacharyya
    distance. This penalizes mismatches in the *relative proportion*
    across tasks at each genomic position, independent of overall scale.

    Shape convention: (B, T, L) where T = tasks and L = output bins.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted signal, shape (B, T, L).
            target: Target signal, same shape.
            mask: Optional mask, shape (B, T, L). Loss is averaged over
                bins where *any* task is valid.
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # Normalize across tasks (dim=1) at each bin
        pred_sum = pred.sum(dim=1, keepdim=True).clamp(min=self.eps)
        target_sum = target.sum(dim=1, keepdim=True).clamp(min=self.eps)

        p = pred / pred_sum    # (B, T, L)
        q = target / target_sum

        # Bhattacharyya coefficient per bin: sum_t(sqrt(p_t * q_t))
        bc = torch.sqrt(p * q + self.eps).sum(dim=1)  # (B, L)

        # Distance per bin = -log(BC)
        dist = -torch.log(bc.clamp(min=self.eps))  # (B, L)

        if mask is not None:
            # A bin is valid if any task has coverage
            bin_valid = mask.any(dim=1).float()  # (B, L)
            n_valid = bin_valid.sum().clamp(min=1)
            return (dist * bin_valid).sum() / n_valid

        return dist.mean()


class MaskedMSELoss(nn.Module):
    """
    MSE loss with per-bin masking for multitask genomic prediction.

    Only computes loss on bins where the mask is nonzero, allowing
    training on partially labeled data (e.g. cell types with different
    genomic coverage).
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions, shape (B, T, L).
            target: Targets, same shape.
            mask: Binary mask (1 = valid), same shape.
        """
        sq_err = (pred - target) ** 2 * mask
        n_valid = mask.sum().clamp(min=1)
        return sq_err.sum() / n_valid


class MaskedPoissonLoss(nn.Module):
    """
    Poisson negative log-likelihood loss with masking.

    Appropriate for count-based targets (e.g. RNA-seq read coverage
    from BigWig tracks).

    Args:
        eps: Added to predictions for numerical stability.
        log_input: If True, pred is treated as log-counts.
    """

    def __init__(self, eps: float = 1e-6, log_input: bool = False):
        super().__init__()
        self.eps = eps
        self.log_input = log_input

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.log_input:
            loss = torch.exp(pred) - target * pred
        else:
            loss = pred + self.eps - target * torch.log(pred + self.eps)

        if mask is not None:
            loss = loss * mask
            n_valid = mask.sum().clamp(min=1)
            return loss.sum() / n_valid

        return loss.mean()
