"""Unit tests for the differentiable Mandel-Agol transit model."""

from __future__ import annotations

import torch

from astromap.mandel_agol import (
    differentiable_transit_loss,
    mandel_agol_transit,
)


def test_mandel_agol_shapes_and_values() -> None:
    time = torch.linspace(0.0, 10.0, 500)
    p = torch.tensor(0.10)  # R_p/R_* = 0.1 -> depth ~ 0.01
    a_over_r = torch.tensor(15.0)
    inc = torch.tensor(90.0)
    period = torch.tensor(3.5)
    t0 = torch.tensor(1.75)

    flux = mandel_agol_transit(time, p, a_over_r, inc, period, t0)
    assert flux.shape == (500,)
    assert torch.all(torch.isfinite(flux))

    # Out of transit flux should be close to 1.0 (folding on period)
    dt_folded = torch.abs((time - t0 + period / 2.0) % period - period / 2.0)
    out_of_transit = dt_folded > 0.4
    assert torch.allclose(flux[out_of_transit], torch.tensor(1.0), atol=1e-3)

    # In transit flux should drop near 1 - p^2
    in_transit_center = torch.argmin(flux)
    assert flux[in_transit_center] < 0.992


def test_mandel_agol_autograd_gradients() -> None:
    time = torch.linspace(0.0, 5.0, 200)
    p = torch.tensor(0.12, requires_grad=True)
    a_over_r = torch.tensor(12.0, requires_grad=True)
    inc = torch.tensor(89.0, requires_grad=True)
    period = torch.tensor(2.5, requires_grad=True)
    t0 = torch.tensor(1.25, requires_grad=True)

    flux = mandel_agol_transit(time, p, a_over_r, inc, period, t0)
    loss = torch.sum((flux - 0.98) ** 2)
    loss.backward()

    assert p.grad is not None and torch.isfinite(p.grad)
    assert a_over_r.grad is not None and torch.isfinite(a_over_r.grad)
    assert period.grad is not None and torch.isfinite(period.grad)


def test_differentiable_transit_loss() -> None:
    time = torch.linspace(0.0, 10.0, 256)
    obs_flux = torch.ones(256)
    obs_flux[120:136] = 0.985

    pred_depth = torch.tensor(0.015, requires_grad=True)
    pred_dur = torch.tensor(0.15, requires_grad=True)
    pred_per = torch.tensor(3.2, requires_grad=True)

    loss = differentiable_transit_loss(
        time=time,
        observed_flux=obs_flux,
        predicted_depth=pred_depth,
        predicted_duration_days=pred_dur,
        predicted_period_days=pred_per,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert pred_depth.grad is not None and torch.isfinite(pred_depth.grad)

