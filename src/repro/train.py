"""Train ResNet-18 on CIFAR-10 with one seed.

    python -m repro.train --config src/repro/configs/resnet18_cifar10.yaml --seed 0 \
        --out results/per_seed/seed_0

Writes to ``--out``:
    train_log.csv   one row per optimizer step (columns in LOG_COLUMNS). val_* are filled on
                    the last step of each epoch and left blank elsewhere.
    meta.json       environment, config, and timing
    model_final.pt  final-epoch weights (not committed)
    eval.json       metrics for model_final.pt (see repro.eval)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from repro.data import CifarArrays, DeviceLoader, apply_subsets, load_cifar10
from repro.eval import evaluate_checkpoint, predict
from repro.model import build_model, count_params

LOG_COLUMNS = ["epoch", "step", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "wall_time"]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    s = cfg["smoke"]
    cfg["train"]["epochs"] = s["epochs"]
    cfg["data"]["train_subset"] = s["train_subset"]
    cfg["data"]["test_subset"] = s["test_subset"]
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (CPU + all GPUs), and pin cuDNN to deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only; recorded for completeness
    # Must be set before the first cuBLAS call for deterministic GEMMs on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def cosine_lr(epoch: int, base_lr: float, epochs: int, min_lr: float = 0.0) -> float:
    """Closed form of CosineAnnealingLR(T_max=epochs) stepped once per epoch (epoch is 0-based)."""
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast_dtype(cfg: dict, device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and cfg["train"].get("precision") == "bf16":
        return torch.bfloat16
    return None


@contextlib.contextmanager
def keep_awake():
    """Stop Windows from sleeping mid-run. Lasts only while the process lives; no settings change."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def environment_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _read_log(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_log(path: Path, rows: list[dict], mode: str) -> None:
    new = mode == "w" or not path.exists()
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def _fmt(v: float | None, spec: str) -> str:
    return "" if v is None else format(v, spec)


def run_training(
    cfg: dict,
    seed: int,
    data: CifarArrays,
    device: torch.device,
    out_dir: Path | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one model. Returns {"rows", "losses" (raw per-step floats), "val_acc", "model", "wall_time_s"}.

    ``max_steps`` stops early (used by the determinism test). With ``out_dir``, the per-step log,
    a resumable checkpoint, and the final weights are written there.
    """
    tcfg, dcfg, ocfg = cfg["train"], cfg["data"], cfg["optim"]
    set_seed(seed, tcfg.get("deterministic", True))
    epochs = tcfg["epochs"]

    model = build_model(cfg).to(device)
    channels_last = bool(tcfg.get("channels_last")) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=ocfg["lr"],
        momentum=ocfg["momentum"],
        weight_decay=ocfg["weight_decay"],
        nesterov=ocfg.get("nesterov", False),
    )
    amp = autocast_dtype(cfg, device)

    train_loader = DeviceLoader(
        data.train_x, data.train_y, tcfg["batch_size"], dcfg["mean"], dcfg["std"],
        device=device, train=True, pad=dcfg["random_crop_padding"], hflip=dcfg["hflip"], seed=seed,
    )
    val_loader = DeviceLoader(
        data.test_x, data.test_y, tcfg["eval_batch_size"], dcfg["mean"], dcfg["std"], device=device,
    )

    start_epoch, elapsed_before = 0, 0.0
    log_path = ckpt_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path, ckpt_path = out_dir / "train_log.csv", out_dir / "ckpt_last.pt"
        if resume and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch, elapsed_before = ckpt["epoch"], ckpt["elapsed"]
            kept = [r for r in _read_log(log_path) if int(r["epoch"]) <= start_epoch]
            _write_log(log_path, kept, "w")
            log(f"resumed from epoch {start_epoch} ({elapsed_before / 60:.1f} min already spent)")
        else:
            _write_log(log_path, [], "w")

    all_rows: list[dict] = []
    all_losses: list[float] = []
    step = start_epoch * len(train_loader)
    val_acc = None
    t0 = time.perf_counter() - elapsed_before
    for epoch in range(start_epoch, epochs):
        lr = cosine_lr(epoch, ocfg["lr"], epochs, cfg["schedule"].get("min_lr", 0.0))
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        losses, corrects, sizes, stamps = [], [], [], []
        for xb, yb in train_loader.epoch(epoch):
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Keep stats on-device and read them back once per epoch, so the GPU never waits for
            # a host sync. wall_time is therefore a host-side timestamp, and is exact only on the
            # last row of each epoch (taken after a synchronising eval).
            losses.append(loss.detach().float())
            corrects.append((logits.argmax(1) == yb).sum())
            sizes.append(len(yb))
            stamps.append(time.perf_counter() - t0)
            step += 1
            if max_steps is not None and step >= max_steps:
                break

        epoch_done = len(sizes) == len(train_loader)
        val_loss = None
        if epoch_done:
            # fp32 eval, matching eval.json; bf16 is only used for the training forward/backward.
            logits_v, labels_v = predict(model, val_loader, None, channels_last)
            val_loss = F.cross_entropy(logits_v, labels_v).item()
            val_acc = (logits_v.argmax(1) == labels_v).float().mean().item() * 100.0
            stamps[-1] = time.perf_counter() - t0

        loss_vals = torch.stack(losses).tolist()
        acc_vals = (torch.stack(corrects).float().cpu() / torch.tensor(sizes) * 100.0).tolist()
        first = step - len(sizes) + 1
        rows = []
        for i, (lv, av, ts) in enumerate(zip(loss_vals, acc_vals, stamps)):
            last = epoch_done and i == len(sizes) - 1
            rows.append({
                "epoch": epoch + 1,
                "step": first + i,
                "lr": _fmt(lr, ".6g"),
                "train_loss": _fmt(lv, ".6g"),
                "train_acc": _fmt(av, ".4f"),
                "val_loss": _fmt(val_loss if last else None, ".6g"),
                "val_acc": _fmt(val_acc if last else None, ".4f"),
                "wall_time": _fmt(ts, ".3f"),
            })
        all_rows.extend(rows)
        all_losses.extend(loss_vals)

        if out_dir is not None:
            _write_log(log_path, rows, "a")
        if out_dir is not None and epoch_done and (epoch + 1) % tcfg.get("checkpoint_every", 1) == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "elapsed": stamps[-1]},
                ckpt_path,
            )
        if epoch_done:
            mean_loss = sum(l * n for l, n in zip(loss_vals, sizes)) / sum(sizes)
            mean_acc = sum(a * n for a, n in zip(acc_vals, sizes)) / sum(sizes)
            log(f"epoch {epoch + 1:3d}/{epochs}  lr {lr:.5f}  train_loss {mean_loss:.4f}  "
                f"train_acc {mean_acc:6.2f}  val_loss {val_loss:.4f}  val_acc {val_acc:6.2f}  "
                f"t {stamps[-1] / 60:6.1f} min")
        if max_steps is not None and step >= max_steps:
            break

    total = time.perf_counter() - t0
    if out_dir is not None:
        torch.save({"model": model.state_dict(), "cfg": cfg, "seed": seed, "epoch": epochs},
                   out_dir / "model_final.pt")
        ckpt_path.unlink(missing_ok=True)
    return {"rows": all_rows, "losses": all_losses, "val_acc": val_acc, "model": model,
            "wall_time_s": total}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke", action="store_true", help="apply the config's smoke overrides and assert min top-1")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", help="override data.root")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt_last.pt if present")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device = resolve_device(args.device)

    data = load_cifar10(cfg["data"]["root"])
    data = apply_subsets(data, cfg["data"]["train_subset"], cfg["data"]["test_subset"])
    print(f"seed {args.seed} | device {device} | train {len(data.train_y)} | test {len(data.test_y)} | "
          f"epochs {cfg['train']['epochs']} | precision {autocast_dtype(cfg, device) or torch.float32}")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with keep_awake():
        result = run_training(cfg, args.seed, data, device, out_dir=args.out, resume=args.resume)

    metrics = evaluate_checkpoint(args.out / "model_final.pt", data=data, device=device,
                                  out_json=args.out / "eval.json")
    meta = {
        "seed": args.seed,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_wall_time_s": round(result["wall_time_s"], 1),
        "n_params": count_params(result["model"]),
        "environment": environment_info(device),
        "config": cfg,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    print(f"final top-1 {metrics['top1']:.2f}%  top-5 {metrics['top5']:.2f}%  "
          f"train time {result['wall_time_s'] / 60:.1f} min")

    if args.smoke:
        floor = cfg["smoke"]["min_top1"]
        if metrics["top1"] <= floor:
            print(f"SMOKE FAILED: top-1 {metrics['top1']:.2f}% <= {floor}%")
            return 1
        print(f"SMOKE OK: top-1 {metrics['top1']:.2f}% > {floor}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
