"""CIFAR-10 loading and on-device augmentation.

The whole dataset (150 MB as uint8) is kept on the training device, and augmentation is done
there in batches. Randomness (shuffle order, crop offsets, flips) comes from a CPU
``torch.Generator`` seeded from ``(seed, epoch)``, so:
  * the data order is identical on CPU and GPU,
  * epoch ``k`` can be regenerated exactly when resuming, and
  * no DataLoader worker processes are involved (slow to spawn on Windows, and another
    source of nondeterminism).

The augmentation reproduces torchvision's ``RandomCrop(32, padding=4)`` followed by
``RandomHorizontalFlip()`` on raw pixels: zero padding is applied before normalisation, as
torchvision does when it pads a PIL image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


@dataclass
class CifarArrays:
    """Raw CIFAR-10 arrays. Images are uint8 NHWC, labels int64."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def load_cifar10(root: str | Path, download: bool = True) -> CifarArrays:
    # torchvision is used here only to download and parse the dataset.
    from torchvision.datasets import CIFAR10

    train = CIFAR10(str(root), train=True, download=download)
    test = CIFAR10(str(root), train=False, download=download)
    return CifarArrays(
        train_x=np.asarray(train.data, dtype=np.uint8),
        train_y=np.asarray(train.targets, dtype=np.int64),
        test_x=np.asarray(test.data, dtype=np.uint8),
        test_y=np.asarray(test.targets, dtype=np.int64),
    )


def stratified_subset(x: np.ndarray, y: np.ndarray, n: int | None, seed: int = 1234):
    """Take ``n`` examples split evenly across classes, using a fixed seed.

    The subset does not depend on the run seed, so smoke runs with different seeds see the same
    data.
    """
    if n is None or n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = n // len(classes)
    idx = np.concatenate(
        [rng.permutation(np.flatnonzero(y == c))[:per_class] for c in classes]
    )
    idx = np.sort(idx)
    return x[idx], y[idx]


def apply_subsets(data: CifarArrays, n_train: int | None, n_test: int | None) -> CifarArrays:
    tx, ty = stratified_subset(data.train_x, data.train_y, n_train)
    vx, vy = stratified_subset(data.test_x, data.test_y, n_test)
    return CifarArrays(tx, ty, vx, vy)


def compute_mean_std(x_uint8_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over all pixels, in [0, 1] units (one channel at a time to bound memory)."""
    mean, std = np.zeros(3), np.zeros(3)
    for c in range(3):
        ch = x_uint8_nhwc[..., c].astype(np.float32) / 255.0
        mean[c] = ch.mean(dtype=np.float64)
        std[c] = ch.std(dtype=np.float64)
    return mean, std


def epoch_generator(seed: int, epoch: int) -> torch.Generator:
    state = np.random.SeedSequence([seed, epoch]).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator().manual_seed(int(state) & 0x7FFF_FFFF_FFFF_FFFF)


class DeviceLoader:
    """Iterates over ``(images, labels)`` batches that already live on ``device``.

    Images come out as normalized float32 NCHW. With ``train=True`` the order is shuffled and
    random crop + horizontal flip are applied. With ``train=False`` the order is fixed and
    images are unaugmented.
    """

    def __init__(
        self,
        x_uint8_nhwc: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        device: torch.device | str = "cpu",
        train: bool = False,
        pad: int = 4,
        hflip: bool = True,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.train = train
        self.pad = pad if train else 0
        self.hflip = hflip and train
        self.seed = seed
        x = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).permute(0, 3, 1, 2).contiguous()
        self.n, _, self.h, self.w = x.shape
        if self.pad:
            x = F.pad(x, (self.pad,) * 4, mode="constant", value=0)
        self.x = x.to(self.device)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(self.device)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1) * 255.0

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def epoch(self, epoch: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.train:
            g = epoch_generator(self.seed, epoch)
            order = torch.randperm(self.n, generator=g)
            span = 2 * self.pad + 1
            dy = torch.randint(0, span, (self.n,), generator=g)
            dx = torch.randint(0, span, (self.n,), generator=g)
            flip = torch.rand(self.n, generator=g) < 0.5 if self.hflip else torch.zeros(self.n, dtype=torch.bool)
        else:
            order = torch.arange(self.n)

        ar_h = torch.arange(self.h, device=self.device)
        ar_w = torch.arange(self.w, device=self.device)
        ch = torch.arange(3, device=self.device).view(1, 3, 1, 1)
        for start in range(0, self.n, self.batch_size):
            sel = order[start:start + self.batch_size]
            idx = sel.to(self.device)
            xb = self.x[idx]
            if self.train:
                b = len(sel)
                oy = dy[sel].to(self.device)
                ox = dx[sel].to(self.device)
                fl = flip[sel].to(self.device)
                rows = oy[:, None] + ar_h[None, :]
                # A horizontal flip of the crop is the same crop read with reversed columns.
                cols = ox[:, None] + torch.where(fl[:, None], (self.w - 1) - ar_w[None, :], ar_w[None, :])
                bi = torch.arange(b, device=self.device).view(b, 1, 1, 1)
                xb = xb[bi, ch, rows[:, None, :, None], cols[:, None, None, :]]
            xb = (xb.float() - self.mean) / self.std
            yield xb, self.y[idx]
