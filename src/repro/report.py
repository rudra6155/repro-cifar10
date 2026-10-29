"""Aggregate per-seed results, draw the plots, and regenerate the numbers in FINDINGS.md / README.md.

    python -m repro.report --results results --findings FINDINGS.md --readme README.md [--check]

Pipeline: results/per_seed/seed_*/{eval.json,train_log.csv,meta.json} -> results/aggregate.csv
-> generated blocks in the markdown files. The markdown blocks are rendered **only** from
aggregate.csv, never from the per-seed files directly, so the CSV is the single source of
truth. ``--check`` re-renders and exits 1 if the committed markdown has drifted from the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from repro.data import CLASSES

REFERENCE_TOP1 = 93.02           # kuangliu/pytorch-cifar README, ResNet18
COMMONLY_CITED = (94.0, 95.0)
# Two-sided 95% Student-t critical values by degrees of freedom.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

NUMERIC = (
    ["top1", "top5", "test_loss", "best_top1", "best_epoch", "final_train_loss", "final_train_acc",
     "epochs", "train_wall_time_h"]
    + [f"acc_{c}" for c in CLASSES]
)
COLUMNS = ["seed"] + NUMERIC + ["gpu", "torch", "started_utc", "finished_utc"]

# Plot styling (light surface; categorical slots 1-3 validate all-pairs for CVD).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# --------------------------------------------------------------------------- aggregate


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_dirs(results: Path) -> list[Path]:
    dirs = [d for d in (results / "per_seed").glob("seed_*") if (d / "eval.json").exists()]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def epoch_rows(log: list[dict]) -> list[dict]:
    return [r for r in log if r["val_acc"] != ""]


def summarize_seed(d: Path) -> dict:
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    log = _read_csv(d / "train_log.csv")
    per_epoch = epoch_rows(log)
    best = max(per_epoch, key=lambda r: float(r["val_acc"]))
    last_epoch = int(log[-1]["epoch"])
    last = [r for r in log if int(r["epoch"]) == last_epoch]
    row = {
        "seed": int(d.name.split("_")[1]),
        "top1": ev["top1"],
        "top5": ev["top5"],
        "test_loss": ev["loss"],
        "best_top1": float(best["val_acc"]),
        "best_epoch": int(best["epoch"]),
        "final_train_loss": statistics.fmean(float(r["train_loss"]) for r in last),
        "final_train_acc": statistics.fmean(float(r["train_acc"]) for r in last),
        "epochs": last_epoch,
        "train_wall_time_h": meta["train_wall_time_s"] / 3600.0,
        "gpu": meta["environment"].get("gpu", meta["environment"]["device"]),
        "torch": meta["environment"]["torch"],
        "started_utc": meta["started_utc"],
        "finished_utc": meta["finished_utc"],
    }
    for c in CLASSES:
        row[f"acc_{c}"] = ev["per_class_acc"][c]
    return row


def _f(v: float) -> str:
    return f"{v:.4f}"


def write_aggregate(rows: list[dict], path: Path) -> None:
    """Per-seed rows, then ``mean`` and ``std`` (sample std, ddof=1) over seeds."""
    out = []
    for r in rows:
        out.append({k: (_f(r[k]) if isinstance(r[k], float) else r[k]) for k in COLUMNS})
    for label, fn in (("mean", statistics.fmean), ("std", statistics.stdev)):
        agg = {"seed": label}
        for k in NUMERIC:
            vals = [float(r[k]) for r in rows]
            agg[k] = _f(fn(vals)) if len(vals) > 1 or label == "mean" else ""
        out.append(agg)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)


# --------------------------------------------------------------------------- render


def _load_aggregate(path: Path) -> tuple[list[dict], dict, dict]:
    rows = _read_csv(path)
    seeds = [r for r in rows if r["seed"] not in ("mean", "std")]
    mean = next(r for r in rows if r["seed"] == "mean")
    std = next(r for r in rows if r["seed"] == "std")
    return seeds, mean, std


def _pm(mean: dict, std: dict, k: str, nd: int = 2) -> str:
    m = float(mean[k])
    return f"{m:.{nd}f} ± {float(std[k]):.{nd}f}" if std[k] != "" else f"{m:.{nd}f}"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def render_results(agg_path: Path) -> str:
    seeds, mean, std = _load_aggregate(agg_path)
    n = len(seeds)
    lines = [
        "| Run | Top-1 (%) | Top-5 (%) | Test loss | Best-epoch top-1 (%) ¹ | Final train acc (%) ² | Train time (h) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in seeds:
        lines.append(
            f"| seed {s['seed']} | {float(s['top1']):.2f} | {float(s['top5']):.2f} | "
            f"{float(s['test_loss']):.4f} | {float(s['best_top1']):.2f} (ep {s['best_epoch']}) | "
            f"{float(s['final_train_acc']):.2f} | {float(s['train_wall_time_h']):.2f} |"
        )
    lines.append(
        f"| **mean ± std (n={n})** | **{_pm(mean, std, 'top1')}** | {_pm(mean, std, 'top5')} | "
        f"{_pm(mean, std, 'test_loss', 4)} | {_pm(mean, std, 'best_top1')} | "
        f"{_pm(mean, std, 'final_train_acc')} | {_pm(mean, std, 'train_wall_time_h')} |"
    )

    m = float(mean["top1"])
    top1s = [float(s["top1"]) for s in seeds]
    delta = m - REFERENCE_TOP1
    lo, hi = COMMONLY_CITED
    band = "inside" if lo <= m <= hi else ("above" if m > hi else "below")
    bullets = [
        f"- Reference (kuangliu/pytorch-cifar README, ResNet18): **{REFERENCE_TOP1:.2f}%**. "
        f"Δ (our mean − reference) = **{delta:+.2f} pp**.",
        f"- Commonly cited band {lo:.0f}–{hi:.0f}%: our mean is **{band}** it "
        f"({m - hi:+.2f} pp vs the top of the band).",
    ]
    if n > 1:
        sd = float(std["top1"])
        t = T_CRIT.get(n - 1, 1.96)
        half = t * sd / math.sqrt(n)
        bullets.append(
            f"- 95% CI for the mean (Student t, df={n - 1}): [{m - half:.2f}, {m + half:.2f}]. "
            f"Seed range (max − min): {max(top1s) - min(top1s):.2f} pp."
        )
        n_above = sum(v > REFERENCE_TOP1 for v in top1s)
        ref_in_ci = m - half <= REFERENCE_TOP1 <= m + half
        bullets.append(
            f"- **Verdict:** {n_above}/{n} seeds beat the reference. The reference "
            f"{'lies inside' if ref_in_ci else 'lies outside'} our 95% CI, so the gap "
            f"{'could be' if ref_in_ci else 'is not'} explained by seed noise. Worst seed vs reference: "
            f"{min(top1s) - REFERENCE_TOP1:+.2f} pp."
        )
    peek = float(mean["best_top1"]) - m
    bullets.append(
        f"- Picking the best epoch on the test set (what the reference script checkpoints) would add "
        f"**{peek:+.2f} pp** on average. We do not use it as the headline."
    )
    if n < 3:
        bullets.append(f"- ⚠️ Only {n} seed(s) present. This is NOT a valid 3-seed reproduction.")
    notes = [
        "",
        f"¹ Max over the {seeds[0]['epochs']} per-epoch test evaluations: an optimistic, test-set-selected number, "
        "shown only for comparison with the reference script's `best_acc`.",
        "² Mean batch accuracy over the last epoch, on augmented training data in train mode.",
    ]
    return "\n".join(lines + [""] + bullets + notes)


def render_per_class(agg_path: Path) -> str:
    seeds, mean, std = _load_aggregate(agg_path)
    order = sorted(CLASSES, key=lambda c: float(mean[f"acc_{c}"]))
    lines = [f"| Class | Top-1 (%), mean ± std over {len(seeds)} seeds |", "|---|---|"]
    lines += [f"| {c} | {_pm(mean, std, f'acc_{c}')} |" for c in order]
    return "\n".join(lines)


def render_compute(agg_path: Path) -> str:
    seeds, mean, std = _load_aggregate(agg_path)
    gpu_hours = sum(float(s["train_wall_time_h"]) for s in seeds)
    wall = (max(_ts(s["finished_utc"]) for s in seeds) - min(_ts(s["started_utc"]) for s in seeds))
    gpus = sorted({s["gpu"] for s in seeds})
    torches = sorted({s["torch"] for s in seeds})
    return "\n".join([
        f"- Hardware: {', '.join(gpus)} (1 GPU, runs executed sequentially); PyTorch {', '.join(torches)}",
        f"- Total GPU-hours (sum of training time over {len(seeds)} seeds): **{gpu_hours:.2f}**",
        f"- Wall-clock, first seed start to last seed end: **{wall.total_seconds() / 3600:.2f} h**",
        f"- Per seed: {_pm(mean, std, 'train_wall_time_h')} h for {seeds[0]['epochs']} epochs",
    ])


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED:{name} [^>]*-->).*?(<!-- END GENERATED:{name} -->)", re.S
    )
    if not pattern.search(text):
        raise ValueError(f"marker block GENERATED:{name} not found")
    return pattern.sub(lambda mt: f"{mt.group(1)}\n{body}\n{mt.group(2)}", text)


def render_docs(agg_path: Path, findings: Path | None, readme: Path | None) -> dict[Path, str]:
    blocks = {
        "results": render_results(agg_path),
        "per_class": render_per_class(agg_path),
        "compute": render_compute(agg_path),
    }
    out = {}
    for path in (p for p in (findings, readme) if p is not None):
        text = path.read_text(encoding="utf-8")
        for name, body in blocks.items():
            if f"GENERATED:{name} " in text:
                text = replace_block(text, name, body)
        out[path] = text
    return out


# --------------------------------------------------------------------------- plots


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_curves(dirs: list[Path], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    for i, d in enumerate(dirs):
        rows = epoch_rows(_read_csv(d / "train_log.csv"))
        ep = [int(r["epoch"]) for r in rows]
        acc = [float(r["val_acc"]) for r in rows]
        color = SERIES[i % len(SERIES)]
        label = d.name.replace("_", " ")
        for ax in axes:
            ax.plot(ep, acc, color=color, linewidth=1.6, label=label)
        tail = [(e, a) for e, a in zip(ep, acc) if e >= max(ep) - 50]
        axes[1].annotate(f"{label}: {acc[-1]:.2f}", xy=(ep[-1], acc[-1]), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8, color=INK_2)
        axes[1].set_xlim(tail[0][0], max(ep) + 14)
    lo, hi = COMMONLY_CITED
    for ax in axes:
        _style(ax)
        ax.axhspan(lo, hi, color="#cde2fb", alpha=0.45, linewidth=0, zorder=0)
        ax.axhline(REFERENCE_TOP1, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    axes[0].set_ylim(40, 100)
    axes[0].set_ylabel("test top-1 (%)", color=INK_2, fontsize=9)
    axes[0].set_title("Test top-1 per epoch, 3 seeds", loc="left", color=INK, fontsize=11)
    axes[0].text(2, REFERENCE_TOP1 - 2.2, f"reference {REFERENCE_TOP1:.2f}%", color=INK_2, fontsize=8)
    axes[0].text(2, hi + 0.6, f"commonly cited {lo:.0f}–{hi:.0f}%", color=INK_2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    axes[1].set_ylim(REFERENCE_TOP1 - 1, 96.5)
    axes[1].set_title("Last 50 epochs (zoom)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(dirs: list[Path], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    mats = [np.array(json.loads((d / "eval.json").read_text(encoding="utf-8"))["confusion"], float)
            for d in dirs]
    pct = np.mean([m / m.sum(1, keepdims=True) * 100 for m in mats], axis=0)
    # Diagonal (~90%+) would wash out the errors, so colour off-diagonal cells on their own scale.
    off = pct.copy()
    np.fill_diagonal(off, np.nan)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(7.4, 6.4), facecolor=SURFACE)
    vmax = np.nanmax(off)
    im = ax.imshow(off, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(10):
        for j in range(10):
            v = pct[i, j]
            if i == j:
                txt, color, weight = f"{v:.1f}", INK, "bold"
            else:
                txt = f"{v:.1f}" if v >= 0.05 else "·"
                color = "white" if v > 0.55 * vmax else INK_2
                weight = "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(10), CLASSES, fontsize=8, color=INK_2)
    ax.set_xlabel("predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("true", color=INK_2, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix, % of true class, mean over {len(dirs)} seeds\n"
                 "(diagonal = per-class accuracy, bold; colour scales off-diagonal errors only)",
                 loc="left", color=INK, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    cb.set_label("% of true class misclassified as column", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--findings", type=Path, default=Path("FINDINGS.md"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--check", action="store_true",
                   help="do not write anything; exit 1 if the markdown differs from aggregate.csv")
    args = p.parse_args(argv)
    agg = args.results / "aggregate.csv"

    if args.check:
        stale = [str(path) for path, text in render_docs(agg, args.findings, args.readme).items()
                 if path.read_text(encoding="utf-8") != text]
        if stale:
            print(f"OUT OF SYNC with {agg}: {', '.join(stale)}. Run `make report`.")
            return 1
        print(f"markdown in sync with {agg}")
        return 0

    dirs = seed_dirs(args.results)
    if not dirs:
        print(f"no completed seeds under {args.results / 'per_seed'}")
        return 1
    write_aggregate([summarize_seed(d) for d in dirs], agg)
    plot_curves(dirs, args.results / "accuracy_curve.png")
    plot_confusion(dirs, args.results / "confusion.png")
    for path, text in render_docs(agg, args.findings, args.readme).items():
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {agg}, plots, and regenerated tables in {args.findings} / {args.readme} "
          f"({len(dirs)} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
