"""Tests for training.multitask_head."""

import torch
import pytest

from training.multitask_head import (
    NonlinearHead,
    SplitHead,
    ConditionalHead,
    CalibratedHeadWrapper,
)


B, C, L = 4, 64, 50  # batch, channels, length


class TestNonlinearHead:
    def test_output_shape(self):
        head = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        x = torch.randn(B, C, L)
        assert head(x).shape == (B, 3, L)

    def test_gradient_flows(self):
        head = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        x = torch.randn(B, C, L, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None


class TestSplitHead:
    def test_output_shape(self):
        head = SplitHead(in_channels=C, hidden=32, task_hidden=16, out_channels=3)
        x = torch.randn(B, C, L)
        assert head(x).shape == (B, 3, L)

    def test_tasks_produce_different_outputs(self):
        head = SplitHead(in_channels=C, hidden=32, task_hidden=16, out_channels=3)
        x = torch.randn(1, C, L)
        out = head(x)
        # Different task heads should generally produce different output
        # (not guaranteed but extremely unlikely with random weights)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_single_task(self):
        head = SplitHead(in_channels=C, hidden=32, task_hidden=16, out_channels=1)
        x = torch.randn(B, C, L)
        assert head(x).shape == (B, 1, L)


class TestConditionalHead:
    def test_output_shape(self):
        head = ConditionalHead(in_channels=C, hidden=32, n_celltypes=5, out_channels=1)
        x = torch.randn(B, C, L)
        ids = torch.tensor([0, 1, 2, 3])
        assert head(x, ids).shape == (B, 1, L)

    def test_different_celltypes_different_output(self):
        head = ConditionalHead(in_channels=C, hidden=32, n_celltypes=5, out_channels=1)
        x = torch.randn(1, C, L)
        out0 = head(x, torch.tensor([0]))
        out3 = head(x, torch.tensor([3]))
        assert not torch.allclose(out0, out3)

    def test_scalar_celltype_id(self):
        head = ConditionalHead(in_channels=C, hidden=32, n_celltypes=5, out_channels=1)
        x = torch.randn(1, C, L)
        out = head(x, torch.tensor(2))  # scalar, not 1D
        assert out.shape == (1, 1, L)

    def test_multichannel_output(self):
        head = ConditionalHead(in_channels=C, hidden=32, n_celltypes=3, out_channels=4)
        x = torch.randn(B, C, L)
        ids = torch.zeros(B, dtype=torch.long)
        assert head(x, ids).shape == (B, 4, L)


class TestCalibratedHeadWrapper:
    def test_output_shape(self):
        base = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        cal = CalibratedHeadWrapper(base, n_tasks=3)
        x = torch.randn(B, C, L)
        assert cal(x).shape == (B, 3, L)

    def test_initial_calibration_is_identity(self):
        base = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        cal = CalibratedHeadWrapper(base, n_tasks=3)
        x = torch.randn(B, C, L)
        base_out = base(x)
        cal_out = cal(x)
        torch.testing.assert_close(cal_out, base_out)

    def test_scale_and_bias_affect_output(self):
        base = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        cal = CalibratedHeadWrapper(base, n_tasks=3)
        cal.scale.data.fill_(2.0)
        cal.bias.data.fill_(1.0)
        x = torch.randn(B, C, L)
        base_out = base(x)
        cal_out = cal(x)
        expected = 2.0 * base_out + 1.0
        torch.testing.assert_close(cal_out, expected)

    def test_in_channels_property(self):
        base = NonlinearHead(in_channels=C, hidden=32, out_channels=3)
        cal = CalibratedHeadWrapper(base, n_tasks=3)
        assert cal.in_channels == C
