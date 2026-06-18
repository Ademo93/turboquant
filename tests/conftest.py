"""Shared pytest fixtures: small reproducible models for fast tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class TinyConv(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def tiny_mlp() -> nn.Module:
    torch.manual_seed(0)
    return TinyMLP()


@pytest.fixture
def tiny_conv() -> nn.Module:
    torch.manual_seed(0)
    return TinyConv()


@pytest.fixture
def mlp_input() -> torch.Tensor:
    return torch.randn(4, 32)


@pytest.fixture
def image_input() -> torch.Tensor:
    return torch.randn(2, 3, 16, 16)
