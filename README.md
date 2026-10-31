# repro-cifar10

A 3-seed, end-to-end reproduction of ResNet-18 on CIFAR-10, compared against the
commonly cited ~93–95% top-1. Results, variance, and what didn't match are in [FINDINGS.md](FINDINGS.md).

## What this is / is NOT

**This is**
- A from-scratch ResNet-18 (CIFAR variant: 3x3 stride-1 stem, no max-pool) trained with the
  standard recipe (SGD 0.1 / momentum 0.9 / wd 5e-4, cosine, 200 epochs, batch 128,
  crop + flip), over 3 seeds, reported as mean ± std of the final-epoch test accuracy.
- Seeded and deterministic (cuDNN deterministic kernels; a test checks bit-identical loss across
  two runs).
- A pipeline where the FINDINGS results table is generated from `results/aggregate.csv`,
  never typed by hand (a test enforces this).

**This is NOT**
- A reproduction of a number from the ResNet paper itself. He et al. 2016 never report
  ResNet-18 on CIFAR-10. The number comes from `kuangliu/pytorch-cifar` (see FINDINGS).
- A wrapper around `torchvision.models.resnet18`. torchvision's model is only used in a test,
  as a shape/parameter cross-check.
- A hyperparameter search or a state-of-the-art attempt. The recipe is fixed in advance.

## Reproduce in 3 commands

```bash
make install   # .venv + torch (CUDA on Windows/Linux GPU) + package
make test      # unit tests, CPU, < 60 s
make repro     # 3 seeds x 200 epochs -> results/ -> aggregate.csv -> plots -> FINDINGS table
```

Also: `make smoke` (2 epochs on a 10k subset, asserts top-1 > 20%; sanity only), `make docker`
(CPU image that runs test + smoke), `make report` (re-aggregate and regenerate the tables
without retraining), and `RESUME=1 make repro` (continue after an interruption; checkpoints
every 10 epochs).

On Windows, run `make` from Git Bash (the recipes use bash). CIFAR-10 (~170 MB) is
downloaded to `data/` on first use.

## Hardware and compute (generated from `results/aggregate.csv`)

<!-- BEGIN GENERATED:compute (source: results/aggregate.csv; regenerate with `make report`) -->
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (1 GPU, runs executed sequentially); PyTorch 2.14.1+cu126
- Total GPU-hours (sum of training time over 3 seeds): **2.60**
- Wall-clock, first seed start to last seed end: **2.62 h**
- Per seed: 0.87 ± 0.02 h for 200 epochs
<!-- END GENERATED:compute -->

## Layout

```
src/repro/model.py      ResNet-18 (CIFAR variant), written from scratch
src/repro/data.py       CIFAR-10 loading + on-device crop/flip augmentation (seeded per epoch)
src/repro/train.py      training loop, per-step CSV log, seeding, resumable checkpoints
src/repro/eval.py       top-1/top-5/per-class/confusion for a checkpoint -> eval.json
src/repro/report.py     aggregate.csv, plots, and regenerating the FINDINGS/README blocks
src/repro/configs/      the recipe (single YAML)
results/per_seed/seed_N/{train_log.csv, eval.json, meta.json}
results/{aggregate.csv, accuracy_curve.png, confusion.png}
```

`train_log.csv` has one row per optimizer step:
`epoch, step, lr, train_loss, train_acc, val_loss, val_acc, wall_time`. `train_loss`/`train_acc`
are for that mini-batch (augmented, train mode). `val_*` are filled on the last step of each
epoch and blank elsewhere. `wall_time` is seconds since the start of training, read on the host
without a per-step GPU sync, so it is exact only on epoch-end rows. "val" is the CIFAR-10 test
set, as in the reference (no hold-out split). It is never used to pick an epoch or a
hyperparameter.

## Known limitations

- 3 seeds is a small sample. The 95% CI on the mean (in FINDINGS) is wide because the t
  critical value at df=2 is 4.3.
- bf16 autocast is used for training speed. The reference trains in fp32. Not ablated (see
  FINDINGS).
- No separate validation split. We report the final epoch, fixed in advance, so the test set
  is not used for selection. The per-epoch test curve is still shown.
- Single machine, single GPU type. cuDNN determinism holds per machine/driver/library
  version. A different GPU will give slightly different (equally valid) numbers.
- Model checkpoints (`*.pt`) and the dataset are not committed.
