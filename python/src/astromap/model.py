"""Compact dual-domain network and physics-aware objective."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class PinnTransNet(nn.Module):
    """Fuse temporal and wavelet features for classification and regression."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 48, kernel_size=7, padding=3, dilation=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.wavelet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fusion = nn.Sequential(nn.Linear(48 * 16 + 24 * 4 * 8, 128), nn.GELU())
        self.classifier = nn.Linear(128, 1)
        self.regressor = nn.Linear(128, 3)

    def forward(self, flux_and_mask: Tensor, scalogram: Tensor) -> tuple[Tensor, Tensor]:
        temporal = self.temporal(flux_and_mask).flatten(1)
        wavelet = self.wavelet(scalogram).flatten(1)
        features = self.fusion(torch.cat((temporal, wavelet), dim=1))
        logits = self.classifier(features).squeeze(1)
        positive_parameters = functional.softplus(self.regressor(features))
        return logits, positive_parameters


def physics_loss(
    parameters: Tensor,
    stellar_radius_solar: Tensor,
    stellar_mass_solar: Tensor,
) -> Tensor:
    """Penalize transit parameters that imply geometrically impossible durations.

    Parameters are depth, duration in days and period in days. The duration
    ceiling assumes a central circular transit and therefore forms a conservative
    upper bound rather than claiming unique orbital reconstruction.
    """

    depth, duration, period = parameters.unbind(dim=1)
    radius_ratio = torch.sqrt(torch.clamp(depth, min=1e-8, max=0.25))
    period_seconds = torch.clamp(period, min=0.05) * 86400.0
    gravitational_constant = 6.67430e-11
    solar_mass = 1.98847e30
    solar_radius = 6.957e8
    semi_major = (
        gravitational_constant
        * stellar_mass_solar
        * solar_mass
        * period_seconds.square()
        / (4.0 * math.pi**2)
    ).pow(1.0 / 3.0)
    ratio = torch.clamp(
        stellar_radius_solar * solar_radius * (1.0 + radius_ratio) / semi_major,
        min=0.0,
        max=0.999,
    )
    max_duration = period * torch.asin(ratio) / math.pi
    duration_penalty = functional.relu(duration - max_duration).square()
    ordering_penalty = functional.relu(duration - period).square()
    return (duration_penalty + ordering_penalty).mean()
