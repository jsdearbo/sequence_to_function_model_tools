"""Tests for training.losses."""

import torch
import pytest

from training.losses import PSILoss, BhattacharyyaLoss, MaskedMSELoss, MaskedPoissonLoss


class TestPSILoss:
    def test_output_is_scalar(self):
        loss_fn = PSILoss()
        pred = torch.sigmoid(torch.randn(2, 3, 10))
        target = torch.rand(2, 3, 10)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(pred, target, mask)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = PSILoss()
        pred = torch.randn(2, 3, 10, requires_grad=True)
        target = torch.rand(2, 3, 10)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(pred.sigmoid(), target, mask)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_perfect_prediction_is_low(self):
        loss_fn = PSILoss(lambda_mid=0.0)  # disable midpoint weighting
        target = torch.tensor([[[0.9, 0.1, 0.5]]])
        pred = target.clone()
        mask = torch.ones_like(target)
        loss = loss_fn(pred, target, mask)
        assert loss.item() < 1.0

    def test_midpoint_weighting_increases_loss_at_boundary(self):
        loss_fn = PSILoss(lambda_mid=25.0)
        # Error at PSI=0.5 should be weighted higher than at PSI=0.0
        pred = torch.tensor([[[0.7]]])
        target_mid = torch.tensor([[[0.5]]])
        target_edge = torch.tensor([[[0.0]]])
        mask = torch.ones(1, 1, 1)
        loss_mid = loss_fn(pred, target_mid, mask)
        loss_edge = loss_fn(pred, target_edge, mask)
        assert loss_mid.item() > loss_edge.item()

    def test_all_zero_mask_returns_zero(self):
        loss_fn = PSILoss()
        pred = torch.rand(2, 3, 10)
        target = torch.rand(2, 3, 10)
        mask = torch.zeros(2, 3, 10)
        loss = loss_fn(pred, target, mask)
        assert loss.item() == 0.0

    def test_without_mask(self):
        loss_fn = PSILoss(use_mask=False)
        pred = torch.sigmoid(torch.randn(2, 3, 10))
        target = torch.rand(2, 3, 10)
        loss = loss_fn(pred, target, None)
        assert loss.dim() == 0 and not torch.isnan(loss)


class TestBhattacharyyaLoss:
    def test_output_is_scalar(self):
        loss_fn = BhattacharyyaLoss()
        pred = torch.abs(torch.randn(2, 3, 20))
        target = torch.abs(torch.randn(2, 3, 20))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_identical_distributions_low_loss(self):
        loss_fn = BhattacharyyaLoss()
        signal = torch.abs(torch.randn(2, 1, 50)) + 0.1
        loss = loss_fn(signal, signal.clone())
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        loss_fn = BhattacharyyaLoss()
        raw = torch.randn(2, 3, 20, requires_grad=True)
        pred = torch.abs(raw) + 0.1
        target = torch.abs(torch.randn(2, 3, 20))
        loss = loss_fn(pred, target)
        loss.backward()
        assert raw.grad is not None

    def test_with_mask(self):
        loss_fn = BhattacharyyaLoss()
        pred = torch.abs(torch.randn(2, 3, 20))
        target = torch.abs(torch.randn(2, 3, 20))
        mask = torch.ones(2, 3, 20)
        mask[:, :, 10:] = 0
        loss = loss_fn(pred, target, mask)
        assert loss.dim() == 0


class TestMaskedMSELoss:
    def test_zero_on_perfect_prediction(self):
        loss_fn = MaskedMSELoss()
        x = torch.randn(2, 3, 10)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(x, x.clone(), mask)
        assert abs(loss.item()) < 1e-6

    def test_positive_on_different_inputs(self):
        loss_fn = MaskedMSELoss()
        pred = torch.zeros(2, 3, 10)
        target = torch.ones(2, 3, 10)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(pred, target, mask)
        assert loss.item() > 0

    def test_mask_reduces_loss(self):
        loss_fn = MaskedMSELoss()
        pred = torch.zeros(1, 1, 4)
        target = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        mask_all = torch.ones(1, 1, 4)
        mask_half = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        loss_all = loss_fn(pred, target, mask_all)
        loss_half = loss_fn(pred, target, mask_half)
        assert loss_half.item() < loss_all.item()

    def test_all_zero_mask(self):
        loss_fn = MaskedMSELoss()
        loss = loss_fn(torch.randn(2, 3, 10), torch.randn(2, 3, 10), torch.zeros(2, 3, 10))
        assert loss.item() == 0.0


class TestMaskedPoissonLoss:
    def test_output_is_scalar(self):
        loss_fn = MaskedPoissonLoss()
        pred = torch.abs(torch.randn(2, 3, 10)) + 0.1
        target = torch.poisson(torch.ones(2, 3, 10) * 3)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(pred, target, mask)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = MaskedPoissonLoss()
        raw = torch.randn(2, 3, 10, requires_grad=True)
        pred = torch.abs(raw) + 1.0
        target = torch.poisson(torch.ones(2, 3, 10) * 3)
        mask = torch.ones(2, 3, 10)
        loss = loss_fn(pred, target, mask)
        loss.backward()
        assert raw.grad is not None

    def test_log_input_mode(self):
        loss_fn = MaskedPoissonLoss(log_input=True)
        pred = torch.randn(2, 3, 10, requires_grad=True)
        target = torch.poisson(torch.ones(2, 3, 10) * 3)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_without_mask(self):
        loss_fn = MaskedPoissonLoss()
        pred = torch.abs(torch.randn(2, 3, 10)) + 0.1
        target = torch.poisson(torch.ones(2, 3, 10) * 3)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
