"""
Fine-tuning utilities including LoRA injection for Borzoi-class models.

Provides a lightweight LoRA implementation (adapters for Linear and Conv1d)
with model injection, head replacement, and weight merging for
zero-overhead inference.

Designed for parameter-efficient fine-tuning of large genomic foundation
models such as Borzoi, where updating all 300M+ parameters is impractical.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapter injection.

    Args:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor. If None, defaults to ``rank``.
        target_patterns: Glob patterns matching module names to wrap.
            Defaults target the conv tower, transformer blocks, unet,
            and penultimate layers of Borzoi.
        freeze_norms: If True, set BN/LN to eval mode and freeze params.
        verbose: Print wrapped module names during injection.
    """
    rank: int = 8
    alpha: Optional[float] = None
    target_patterns: Tuple[str, ...] = (
        # Conv tower
        "embedding.conv_tower.blocks.*.conv",
        # Transformer self-attention
        "embedding.transformer_tower.blocks.*.mha.to_q",
        "embedding.transformer_tower.blocks.*.mha.to_v",
        # Transformer FFN
        "embedding.transformer_tower.blocks.*.ffn.dense1.linear",
        "embedding.transformer_tower.blocks.*.ffn.dense2.linear",
        # U-net
        "embedding.unet_tower.blocks.*.conv.conv",
        "embedding.unet_tower.blocks.*.channel_transform.conv.layer",
        "embedding.unet_tower.blocks.*.sconv.pointwise",
        # Penultimate bottleneck
        "embedding.pointwise_conv.conv",
    )
    freeze_norms: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# LoRA adapter modules
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    LoRA adapter for nn.Linear.

    Adds a low-rank update: ``y = W_base @ x + (alpha/r) * A @ B @ x``
    where B: in→r and A: r→out, with B Kaiming-initialized and A zero-initialized.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = float(alpha if alpha is not None else r)

        self.B = nn.Linear(base.in_features, r, bias=False)
        self.A = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.B.weight, a=5**0.5)
        nn.init.zeros_(self.A.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def merge_(self):
        """Fold LoRA parameters into the base weight for zero-overhead inference."""
        delta = (self.alpha / self.r) * (self.A.weight @ self.B.weight)
        self.base.weight += delta
        self.A.weight.zero_()
        self.B.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.alpha / self.r) * self.A(self.B(x))


class LoRAConv1d(nn.Module):
    """
    LoRA adapter for nn.Conv1d (groups=1 only).

    Decomposes the update into B: (in_ch, kernel) → r channels and
    A: r → out_ch with a 1×1 convolution.

    Skips depthwise/grouped convolutions by design — these should not
    be wrapped.
    """

    def __init__(self, base: nn.Conv1d, r: int, alpha: Optional[float] = None):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv1d only supports groups=1.")

        self.base = base
        self.r = r
        self.alpha = float(alpha if alpha is not None else r)

        cin = base.in_channels
        cout = base.out_channels

        def _first(v):
            return v[0] if isinstance(v, (tuple, list)) else v

        k = _first(base.kernel_size)
        pad = _first(base.padding)
        dil = _first(base.dilation)
        stride = _first(base.stride)

        self.B = nn.Conv1d(
            cin, r, kernel_size=k, stride=stride,
            padding=pad, dilation=dil, bias=False,
        )
        self.A = nn.Conv1d(r, cout, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.B.weight, a=5**0.5)
        nn.init.zeros_(self.A.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def merge_(self):
        """Fold LoRA parameters into the base weight."""
        A_w = self.A.weight.squeeze(-1)  # [cout, r]
        B_w = self.B.weight              # [r, cin, k]
        delta = torch.einsum("or,rik->oik", A_w, B_w) * (self.alpha / self.r)
        self.base.weight += delta
        self.A.weight.zero_()
        self.B.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.alpha / self.r) * self.A(self.B(x))


# ---------------------------------------------------------------------------
# Injection and merging
# ---------------------------------------------------------------------------

def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _is_depthwise(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv1d) and (m.groups == m.in_channels == m.out_channels)


def _get_parent(root: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


def inject_lora(root: nn.Module, cfg: LoRAConfig) -> None:
    """
    In-place: replace target modules with LoRA-wrapped versions and freeze
    all non-LoRA parameters.

    Args:
        root: The model to modify (e.g. the Borzoi backbone).
        cfg: LoRA configuration specifying rank, targets, etc.
    """
    wrapped_count = 0

    for name, module in list(root.named_modules()):
        if not _name_matches(name, cfg.target_patterns):
            continue

        parent = _get_parent(root, name)
        child_name = name.split(".")[-1]
        wrapped = None

        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(module, r=cfg.rank, alpha=cfg.alpha)
        elif isinstance(module, nn.Conv1d):
            if _is_depthwise(module) or module.groups != 1:
                if cfg.verbose:
                    logger.info("[LoRA] Skipping grouped/depthwise: %s", name)
                continue
            wrapped = LoRAConv1d(module, r=cfg.rank, alpha=cfg.alpha)

        if wrapped is not None:
            setattr(parent, child_name, wrapped)
            wrapped_count += 1
            if cfg.verbose:
                logger.info("[LoRA] Wrapped: %s", name)

    # Freeze everything
    for p in root.parameters():
        p.requires_grad = False

    # Enable LoRA params
    for m in root.modules():
        if isinstance(m, (LoRALinear, LoRAConv1d)):
            m.A.weight.requires_grad = True
            m.B.weight.requires_grad = True

    # Freeze normalization layers
    if cfg.freeze_norms:
        for m in root.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    logger.info("[LoRA] Wrapped %d modules (rank=%d)", wrapped_count, cfg.rank)
    
    if wrapped_count == 0:
        raise RuntimeError(
            "No modules matched LoRA target patterns. "
            "Verify module names with: list(model.named_modules())."
        )


def merge_lora(root: nn.Module) -> None:
    """
    In-place: fold all LoRA adapter weights back into base layers.

    After merging, the model produces identical outputs but with
    zero runtime overhead from the adapters.
    """
    count = 0
    for m in root.modules():
        if isinstance(m, (LoRALinear, LoRAConv1d)):
            m.merge_()
            count += 1
    logger.info("[LoRA] Merged %d adapters into base weights.", count)


def count_trainable_params(model: nn.Module) -> dict:
    """
    Count trainable vs total parameters. Useful for verifying LoRA injection.

    Returns:
        Dict with keys 'trainable', 'total', 'pct_trainable'.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "pct_trainable": 100.0 * trainable / total if total > 0 else 0.0,
    }
