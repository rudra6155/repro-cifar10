"""Guard the published recipe: the config must say what SPEC.md / FINDINGS.md say, and the LR schedule
must equal PyTorch's CosineAnnealingLR as used by the reference script."""

from pathlib import Path

import pytest
import torch

from repro.train import cosine_lr, load_config

CFG = load_config(Path(__file__).parents[1] / "src/repro/configs/resnet18_cifar10.yaml")


def test_recipe_values():
    assert CFG["model"]["arch"] == "resnet18"
    assert CFG["optim"] == {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False}
    assert CFG["schedule"]["name"] == "cosine"
    assert CFG["train"]["epochs"] == 200 and CFG["train"]["batch_size"] == 128
    assert CFG["data"]["random_crop_padding"] == 4 and CFG["data"]["hflip"] is True
    assert CFG["data"]["train_subset"] is None and CFG["data"]["test_subset"] is None


def test_cosine_matches_torch_cosine_annealing_lr():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    for epoch in range(200):
        assert cosine_lr(epoch, 0.1, 200) == pytest.approx(opt.param_groups[0]["lr"], abs=1e-12)
        opt.step()
        sched.step()
    assert cosine_lr(0, 0.1, 200) == 0.1 and cosine_lr(100, 0.1, 200) == pytest.approx(0.05)
