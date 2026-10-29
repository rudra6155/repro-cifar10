"""Evaluate a trained checkpoint on the CIFAR-10 test set.

    python -m repro.eval --ckpt results/per_seed/seed_0/model_final.pt \
        --out results/per_seed/seed_0/eval.json

eval.json holds: top1, top5, loss, per_class_acc {class: %}, confusion (10x10, rows = true class,
cols = predicted), n, checkpoint epoch, and seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from repro.data import CLASSES, CifarArrays, DeviceLoader, apply_subsets, load_cifar10
from repro.model import build_model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DeviceLoader,
    amp: torch.dtype | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (float32 logits, labels) on CPU for every example in ``loader``, in order."""
    model.eval()
    device_type = loader.device.type
    outs, labels = [], []
    for xb, yb in loader.epoch(0):
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type, dtype=amp, enabled=amp is not None):
            outs.append(model(xb).float().cpu())
        labels.append(yb.cpu())
    return torch.cat(outs), torch.cat(labels)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> dict:
    pred = logits.argmax(1)
    top5 = logits.topk(min(5, num_classes), dim=1).indices
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion.index_put_((labels, pred), torch.ones_like(labels), accumulate=True)
    per_class = confusion.diag().float() / confusion.sum(1).clamp(min=1).float() * 100.0
    names = CLASSES if num_classes == len(CLASSES) else [str(i) for i in range(num_classes)]
    return {
        "n": int(labels.numel()),
        "top1": (pred == labels).float().mean().item() * 100.0,
        "top5": (top5 == labels[:, None]).any(1).float().mean().item() * 100.0,
        "loss": F.cross_entropy(logits, labels).item(),
        "per_class_acc": {name: round(v, 2) for name, v in zip(names, per_class.tolist())},
        "confusion": confusion.tolist(),
    }


def evaluate_checkpoint(
    ckpt_path: str | Path,
    data: CifarArrays | None = None,
    device: torch.device | str = "cpu",
    out_json: str | Path | None = None,
) -> dict:
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    if data is None:
        data = load_cifar10(cfg["data"]["root"])
        data = apply_subsets(data, cfg["data"]["train_subset"], cfg["data"]["test_subset"])
    loader = DeviceLoader(
        data.test_x, data.test_y, cfg["train"]["eval_batch_size"],
        cfg["data"]["mean"], cfg["data"]["std"], device=device,
    )
    # Evaluation always runs in fp32, whatever precision training used.
    logits, labels = predict(model, loader)
    metrics = metrics_from_logits(logits, labels, cfg["model"]["num_classes"])
    metrics.update({"seed": ckpt.get("seed"), "epoch": ckpt.get("epoch"), "checkpoint": "final"})
    if out_json is not None:
        Path(out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    m = evaluate_checkpoint(args.ckpt, device=device, out_json=args.out)
    print(f"top-1 {m['top1']:.2f}%  top-5 {m['top5']:.2f}%  loss {m['loss']:.4f}  (n={m['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
