"""Data pipeline: batch shapes, augmentation semantics, ordering, subsets. Uses synthetic data; the
real-CIFAR checks run only if the dataset is already on disk."""

from pathlib import Path

import numpy as np
import pytest
import torch

from repro.data import (
    CLASSES, DeviceLoader, compute_mean_std, load_cifar10, stratified_subset,
)
from repro.train import load_config

CFG = load_config(Path(__file__).parents[1] / "src/repro/configs/resnet18_cifar10.yaml")
MEAN, STD = CFG["data"]["mean"], CFG["data"]["std"]
DATA_ROOT = Path(__file__).parents[1] / CFG["data"]["root"]


def fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, n).astype(np.int64)


def raw(xb):
    """Undo normalisation -> float pixel values in [0, 255]."""
    m = torch.tensor(MEAN).view(1, 3, 1, 1) * 255
    s = torch.tensor(STD).view(1, 3, 1, 1) * 255
    return (xb * s + m).round()


def test_train_batch_shapes_and_partial_last_batch():
    x, y = fake(300)
    dl = DeviceLoader(x, y, 128, MEAN, STD, train=True)
    batches = list(dl.epoch(0))
    assert len(dl) == len(batches) == 3
    assert [b[0].shape for b in batches] == [(128, 3, 32, 32), (128, 3, 32, 32), (44, 3, 32, 32)]
    assert all(b[0].dtype == torch.float32 and b[1].dtype == torch.int64 for b in batches)


def test_each_train_image_is_a_padded_crop_of_its_source_possibly_flipped():
    x, y = fake(64)
    y = np.arange(64)  # labels identify the source image
    dl = DeviceLoader(x, y, 64, MEAN, STD, train=True, pad=4, seed=3)
    xb, yb = next(dl.epoch(0))
    padded = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)))  # zero pad, as torchvision does
    got = raw(xb).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    n_flipped, offsets = 0, set()
    for img, label in zip(got, yb.tolist()):
        matches = [(dy, dx, f) for dy in range(9) for dx in range(9) for f in (0, 1)
                   if np.array_equal(img, (lambda c: c[:, ::-1] if f else c)(padded[label, dy:dy + 32, dx:dx + 32]))]
        assert matches, "output is not a crop(+flip) of its source"
        n_flipped += matches[0][2]
        offsets.add(matches[0][:2])
    assert 10 < n_flipped < 54   # roughly half flipped
    assert len(offsets) > 20     # offsets actually vary


def test_eval_loader_is_unaugmented_and_in_order():
    x, y = fake(100)
    dl = DeviceLoader(x, y, 32, MEAN, STD, train=False)
    xb = torch.cat([b for b, _ in dl.epoch(0)])
    yb = torch.cat([l for _, l in dl.epoch(0)])
    assert torch.equal(raw(xb).permute(0, 2, 3, 1).to(torch.uint8), torch.from_numpy(x))
    assert torch.equal(yb, torch.from_numpy(y))


def test_shuffle_depends_on_seed_and_epoch_only():
    x, y = fake(200)
    y = np.arange(200)
    order = lambda seed, ep: torch.cat([l for _, l in DeviceLoader(x, y, 50, MEAN, STD, train=True, seed=seed).epoch(ep)])
    assert torch.equal(order(0, 5), order(0, 5))
    assert not torch.equal(order(0, 5), order(0, 6))
    assert not torch.equal(order(0, 5), order(1, 5))
    assert sorted(order(0, 5).tolist()) == list(range(200))  # a permutation: each image once per epoch


def test_stratified_subset_is_balanced_and_fixed():
    x, y = fake(5000)
    sx, sy = stratified_subset(x, y, 1000)
    assert len(sy) == 1000 and np.bincount(sy, minlength=10).tolist() == [100] * 10
    assert np.array_equal(sy, stratified_subset(x, y, 1000)[1])


@pytest.mark.skipif(not (DATA_ROOT / "cifar-10-batches-py").exists(), reason="CIFAR-10 not downloaded")
def test_real_cifar10_shapes_and_normalisation_constants():
    d = load_cifar10(DATA_ROOT, download=False)
    assert d.train_x.shape == (50000, 32, 32, 3) and d.test_x.shape == (10000, 32, 32, 3)
    assert np.bincount(d.train_y).tolist() == [5000] * 10 and np.bincount(d.test_y).tolist() == [1000] * 10
    assert len(CLASSES) == 10
    mean, std = compute_mean_std(d.train_x)
    np.testing.assert_allclose(mean, MEAN, atol=1e-3)
    np.testing.assert_allclose(std, STD, atol=1e-3)
