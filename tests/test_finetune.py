"""Tests for training.finetune (LoRA injection, merging, parameter counting)."""

import pytest
import torch
import torch.nn as nn

from training.finetune import (
    LoRAConfig,
    LoRALinear,
    LoRAConv1d,
    inject_lora,
    merge_lora,
    count_trainable_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class ToyModel(nn.Module):
    """Minimal model with Conv1d, Linear, BatchNorm, and depthwise conv."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.linear = nn.Linear(32, 8)
        self.dw = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.bn(x)
        x = x.mean(dim=-1)
        x = self.linear(x)
        x = x.unsqueeze(-1).expand(-1, -1, 5)
        return self.dw(x)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_output_shape_matches_base(self):
        base = nn.Linear(32, 8)
        lora = LoRALinear(base, r=4)
        x = torch.randn(2, 32)
        assert lora(x).shape == (2, 8)

    def test_initial_output_equals_base(self):
        base = nn.Linear(32, 8)
        x = torch.randn(2, 32)
        base_out = base(x).detach()
        lora = LoRALinear(base, r=4)
        lora_out = lora(x).detach()
        # A is zero-initialized, so LoRA adds nothing initially
        torch.testing.assert_close(lora_out, base_out, atol=1e-6, rtol=1e-5)

    def test_merge_preserves_output(self):
        base = nn.Linear(32, 8)
        lora = LoRALinear(base, r=4)
        # Simulate training by setting A to non-zero
        nn.init.normal_(lora.A.weight)
        x = torch.randn(2, 32)
        out_before = lora(x).detach()
        lora.merge_()
        out_after = lora(x).detach()
        torch.testing.assert_close(out_after, out_before, atol=1e-5, rtol=1e-4)

    def test_base_params_frozen(self):
        base = nn.Linear(32, 8)
        lora = LoRALinear(base, r=4)
        assert not lora.base.weight.requires_grad
        assert not lora.base.bias.requires_grad


# ---------------------------------------------------------------------------
# LoRAConv1d
# ---------------------------------------------------------------------------

class TestLoRAConv1d:
    def test_output_shape_matches_base(self):
        base = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        lora = LoRAConv1d(base, r=4)
        x = torch.randn(2, 4, 100)
        assert lora(x).shape == base(x).shape

    def test_initial_output_equals_base(self):
        base = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 4, 50)
        base_out = base(x).detach()
        lora = LoRAConv1d(base, r=4)
        lora_out = lora(x).detach()
        torch.testing.assert_close(lora_out, base_out, atol=1e-6, rtol=1e-5)

    def test_merge_preserves_output(self):
        base = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        lora = LoRAConv1d(base, r=4)
        nn.init.normal_(lora.A.weight)
        x = torch.randn(2, 4, 50)
        out_before = lora(x).detach()
        lora.merge_()
        out_after = lora(x).detach()
        torch.testing.assert_close(out_after, out_before, atol=1e-5, rtol=1e-4)

    def test_rejects_depthwise_conv(self):
        base = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8)
        with pytest.raises(ValueError, match="groups=1"):
            LoRAConv1d(base, r=4)

    def test_base_params_frozen(self):
        base = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        lora = LoRAConv1d(base, r=4)
        assert not lora.base.weight.requires_grad


# ---------------------------------------------------------------------------
# inject_lora / merge_lora / count_trainable_params
# ---------------------------------------------------------------------------

class TestInjectAndMerge:
    def test_inject_wraps_target_layers(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("conv", "linear"), verbose=False)
        inject_lora(model, cfg)
        assert isinstance(model.conv, LoRAConv1d)
        assert isinstance(model.linear, LoRALinear)

    def test_inject_skips_depthwise(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("dw",), verbose=False)
        inject_lora(model, cfg)
        # depthwise should NOT be wrapped
        assert isinstance(model.dw, nn.Conv1d)

    def test_inject_freezes_non_lora_params(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("conv", "linear"), verbose=False)
        inject_lora(model, cfg)
        for name, p in model.named_parameters():
            if "A.weight" in name or "B.weight" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_inject_freezes_batchnorm(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("conv",), freeze_norms=True, verbose=False)
        inject_lora(model, cfg)
        assert not model.bn.weight.requires_grad
        assert not model.bn.bias.requires_grad

    def test_forward_pass_after_inject(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("conv", "linear"), verbose=False)
        inject_lora(model, cfg)
        x = torch.randn(2, 4, 20)
        out = model(x)
        assert out.shape == (2, 8, 5)

    def test_merge_produces_identical_output(self):
        model = ToyModel()
        cfg = LoRAConfig(rank=4, target_patterns=("conv", "linear"), verbose=False)
        inject_lora(model, cfg)
        # Simulate training
        for p in model.parameters():
            if p.requires_grad:
                p.data.normal_()
        model.eval()
        x = torch.randn(1, 4, 20)
        out_before = model(x).detach()
        merge_lora(model)
        out_after = model(x).detach()
        torch.testing.assert_close(out_after, out_before, atol=1e-4, rtol=1e-3)

    def test_count_trainable_params(self):
        model = ToyModel()
        stats_full = count_trainable_params(model)
        assert stats_full["pct_trainable"] == 100.0

        cfg = LoRAConfig(rank=4, target_patterns=("conv", "linear"), verbose=False)
        inject_lora(model, cfg)
        stats_lora = count_trainable_params(model)
        assert stats_lora["pct_trainable"] < 100.0
        assert stats_lora["trainable"] < stats_lora["total"]
