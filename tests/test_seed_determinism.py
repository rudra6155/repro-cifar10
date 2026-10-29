"""Two short training runs with the same seed must produce bit-identical losses (CPU)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from repro.data import CifarArrays
from repro.train import load_config, run_training

CFG = load_config(Path(__file__).parents[1] / "src/repro/configs/resnet18_cifar10.yaml")


def tiny_data():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (320, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, 10, 320).astype(np.int64)
    return CifarArrays(x, y, x[:64], y[:64])


def short_run(seed):
    cfg = copy.deepcopy(CFG)
    cfg["train"].update(epochs=1, batch_size=16)
    return run_training(cfg, seed, tiny_data(), torch.device("cpu"), max_steps=10, log=lambda _: None)


@pytest.fixture(scope="module")
def runs():
    return short_run(0), short_run(0), short_run(1)


def test_same_seed_identical_loss_at_step_10(runs):
    a, b, _ = runs
    assert len(a["losses"]) == len(b["losses"]) == 10
    assert a["losses"][9] == b["losses"][9]          # exact float equality, not approx
    assert a["losses"] == b["losses"]
    assert a["rows"][9]["step"] == 10


def test_same_seed_identical_weights(runs):
    a, b, _ = runs
    for (k, va), vb in zip(a["model"].state_dict().items(), b["model"].state_dict().values()):
        assert torch.equal(va, vb), k


def test_different_seed_differs(runs):
    a, _, c = runs
    assert a["losses"][9] != c["losses"][9]
