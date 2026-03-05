"""
Multitask and conditional head architectures for genomic foundation models.

Provides plug-in prediction heads for Borzoi-class models supporting:
- Simple nonlinear projection
- Split-head with per-task normalization
- Cell-type conditioned prediction via learned embeddings

All heads operate on (B, C, L) feature tensors from the model trunk,
using 1D convolutions to preserve spatial resolution.
"""

import torch
import torch.nn as nn


class NonlinearHead(nn.Module):
    """
    Two-layer nonlinear projection head with GELU activation.

    Maps trunk features to task predictions via Conv1d layers,
    preserving spatial (bin) resolution.

    Args:
        in_channels: Input feature dimension (1920 for Borzoi).
        hidden: Hidden dimension.
        out_channels: Number of output tasks.
    """

    def __init__(self, in_channels: int = 1920, hidden: int = 512, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trunk features, shape (B, C, L).
        Returns:
            Predictions, shape (B, out_channels, L).
        """
        return self.net(x)


class SplitHead(nn.Module):
    """
    Split head: shared nonlinear trunk with task-specific projection networks.

    Each task has its own BatchNorm and nonlinear layers, forcing
    tasks to learn distinct representations. This prevents the
    collapse problem where multitask heads converge to identical
    predictions across tasks.

    Architecture::

        trunk features → shared Conv1d+GELU → [task_0: BN+Conv1d]
                                              [task_1: BN+Conv1d]
                                              ...

    Args:
        in_channels: Input feature dimension.
        hidden: Shared trunk hidden dimension.
        task_hidden: Per-task projection hidden dimension.
        out_channels: Number of tasks.
    """

    def __init__(
        self,
        in_channels: int = 1920,
        hidden: int = 512,
        task_hidden: int = 256,
        out_channels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.shared = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, task_hidden, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(task_hidden),
                nn.Conv1d(task_hidden, 1, kernel_size=1),
            )
            for _ in range(out_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trunk features, shape (B, C, L).
        Returns:
            Predictions, shape (B, out_channels, L).
        """
        shared = self.shared(x)
        outs = [head(shared) for head in self.task_heads]
        return torch.cat(outs, dim=1)


class ConditionalHead(nn.Module):
    """
    Cell-type conditioned prediction head.

    Concatenates a learned cell-type embedding with trunk features
    before projection, enabling a single model to produce cell-type-specific
    predictions. This is more parameter-efficient than separate heads
    per cell type and allows the model to share information across
    related cell types.

    Architecture::

        trunk features (B, C, L) + cell_embed(id) → (B, C+cond_dim, L) → Conv1d → predictions

    Args:
        in_channels: Input feature dimension.
        hidden: Hidden dimension for projection layers.
        cond_dim: Dimensionality of cell-type embeddings.
        n_celltypes: Number of distinct cell types.
        out_channels: Number of output channels per cell type.
    """

    def __init__(
        self,
        in_channels: int = 1920,
        hidden: int = 512,
        cond_dim: int = 32,
        n_celltypes: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cell_embed = nn.Embedding(n_celltypes, cond_dim)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels + cond_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        cell_type_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Trunk features, shape (B, C, L).
            cell_type_id: Integer cell type indices, shape (B,).
        Returns:
            Predictions, shape (B, out_channels, L).
        """
        if cell_type_id.ndim == 0:
            cell_type_id = cell_type_id.unsqueeze(0)

        cond = self.cell_embed(cell_type_id)            # (B, cond_dim)
        cond = cond.unsqueeze(-1)                        # (B, cond_dim, 1)
        cond = cond.expand(-1, -1, features.shape[-1])   # (B, cond_dim, L)

        x = torch.cat([features, cond], dim=1)           # (B, C+cond_dim, L)
        return self.net(x)


class CalibratedHeadWrapper(nn.Module):
    """
    Per-task affine calibration layer applied after any prediction head.

    Learns a scale (α) and bias (β) per task: ``y_calibrated = α * y + β``.
    Useful for aligning pre-trained model outputs to new target distributions
    during fine-tuning.

    Args:
        base_head: The underlying prediction head module.
        n_tasks: Number of output tasks to calibrate.
    """

    def __init__(self, base_head: nn.Module, n_tasks: int):
        super().__init__()
        self.base_head = base_head
        self.scale = nn.Parameter(torch.ones(1, n_tasks, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_tasks, 1))

    @property
    def in_channels(self):
        return self.base_head.in_channels

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.base_head(x, *args, **kwargs)
        return self.scale * y + self.bias
