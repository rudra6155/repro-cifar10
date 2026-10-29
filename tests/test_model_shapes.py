"""Sanity-check our from-scratch ResNet-18 against torchvision's (used here ONLY as a reference)."""

import pytest
import torch
from torch import nn
from torchvision.models import resnet18 as tv_resnet18

from repro.model import count_params, resnet18

# The only structural difference in parameters: 7x7 vs 3x3 stem conv (3 -> 64 channels).
STEM_PARAM_DIFF = 64 * 3 * 7 * 7 - 64 * 3 * 3 * 3
EXPECTED_PARAMS = 11_173_962  # widely quoted figure for CIFAR ResNet-18 with 10 classes


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    return resnet18(num_classes=10).eval(), tv_resnet18(weights=None, num_classes=10).eval()


def test_output_shape(models):
    ours, _ = models
    assert ours(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_param_count_matches_torchvision_minus_stem(models):
    ours, tv = models
    assert count_params(ours) == count_params(tv) - STEM_PARAM_DIFF
    assert count_params(ours) == EXPECTED_PARAMS


def test_state_dict_shapes_match_torchvision_except_stem(models):
    ours, tv = models
    sd_ours, sd_tv = ours.state_dict(), tv.state_dict()
    assert sd_ours.keys() == sd_tv.keys()
    mismatched = [k for k in sd_ours if sd_ours[k].shape != sd_tv[k].shape]
    assert mismatched == ["conv1.weight"]
    assert sd_ours["conv1.weight"].shape == (64, 3, 3, 3)
    assert sd_tv["conv1.weight"].shape == (64, 3, 7, 7)


def test_cifar_stem_has_no_maxpool_and_stride_1(models):
    ours, _ = models
    assert not any(isinstance(m, nn.MaxPool2d) for m in ours.modules())
    assert ours.conv1.stride == (1, 1) and ours.conv1.kernel_size == (3, 3)


def _stage_shapes(model, x):
    shapes = {}
    hooks = [getattr(model, f"layer{i}").register_forward_hook(
        lambda _m, _i, out, i=i: shapes.__setitem__(i, tuple(out.shape[1:]))) for i in range(1, 5)]
    model(x)
    for h in hooks:
        h.remove()
    return shapes


def test_stage_feature_maps(models):
    ours, tv = models
    x = torch.randn(2, 3, 32, 32)
    assert _stage_shapes(ours, x) == {1: (64, 32, 32), 2: (128, 16, 16), 3: (256, 8, 8), 4: (512, 4, 4)}
    # The ImageNet stem would shrink a 32 px input 4x before stage 1, the reason for the CIFAR variant.
    assert _stage_shapes(tv, x) == {1: (64, 8, 8), 2: (128, 4, 4), 3: (256, 2, 2), 4: (512, 1, 1)}


@torch.no_grad()
def test_stages_numerically_match_torchvision_with_shared_weights(models):
    """Run torchvision's weights through our stages; the outputs must agree with torchvision's."""
    ours, tv = models
    sd = {k: v for k, v in tv.state_dict().items() if k != "conv1.weight"}
    ours = resnet18(num_classes=10)
    ours.load_state_dict(sd, strict=False)
    ours.eval()
    x = torch.randn(2, 64, 16, 16)
    a, b = x, x
    for i in range(1, 5):
        a, b = getattr(ours, f"layer{i}")(a), getattr(tv, f"layer{i}")(b)
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(ours.fc(a.mean(dim=(2, 3))), tv.fc(torch.flatten(tv.avgpool(b), 1)))
