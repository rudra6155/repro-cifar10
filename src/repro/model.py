"""ResNet-18 adapted for 32x32 CIFAR inputs, written from scratch.

Differences from the ImageNet ResNet-18 (He et al. 2016, torchvision.models.resnet18):
  * stem is a 3x3 stride-1 conv (not 7x7 stride-2), and
  * there is no max-pool after the stem,
so the four stages see 32/16/8/4 px feature maps instead of 8/4/2/1.

Module names mirror torchvision's (conv1, bn1, layer1..4, downsample, fc) so the test suite can
compare state_dict shapes one-to-one and run torchvision's stage weights through our blocks.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_planes != planes:
            # Projection shortcut (option B in the paper): 1x1 conv + BN.
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, blocks_per_stage: list[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(128, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(256, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(512, blocks_per_stage[3], stride=2)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        # Weights keep PyTorch's default init, like the reference implementation
        # (kuangliu/pytorch-cifar), which also does no custom initialisation.

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        # Global average pool as a mean: equivalent to AdaptiveAvgPool2d(1), and its backward pass
        # is deterministic on CUDA (adaptive_avg_pool2d_backward is not).
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet([2, 2, 2, 2], num_classes=num_classes)


def build_model(cfg: dict) -> ResNet:
    arch = cfg["model"]["arch"]
    if arch != "resnet18":
        raise ValueError(f"unknown arch {arch!r}")
    return resnet18(num_classes=cfg["model"]["num_classes"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
