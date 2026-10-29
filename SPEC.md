# SPEC — acceptance criteria for `repro-cifar10`

This file restates, before any code was written, what "done" means. FINDINGS.md is judged
against it.

## Goal

Reproduce the commonly cited ResNet-18 / CIFAR-10 top-1 accuracy (~93–95%) end-to-end, from a
fresh clone, with **3 seeds** and an honest account of the variance and of everything that did
not match. Being honest matters more than matching the number. "Matched to within X pp across
3 seeds, and here is what still diverged" is the target output.

## Reference being reproduced

- Architecture: He, Zhang, Ren, Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016,
  arXiv:1512.03385. Note that the paper itself does **not** report ResNet-18 on CIFAR-10. Its
  CIFAR experiments use the 3-stage ResNet-20/32/44/56/110 family.
- Number: the ~93–95% figure for "ResNet-18 on CIFAR-10" comes from the community
  implementation `kuangliu/pytorch-cifar` (README: **93.02%**). Its current `main.py` recipe
  is widely reported to reach ~95%. FINDINGS.md has to say this plainly and not dress it up as
  a paper number.

## Fixed recipe (stated in the config AND in FINDINGS.md)

| Item | Value |
|---|---|
| Model | ResNet-18, CIFAR variant: 3x3 stride-1 stem, **no** max-pool, 4 stages of BasicBlock [2,2,2,2], widths 64/128/256/512, global avg-pool, linear 10 |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 5e-4 (on all params) |
| Schedule | Cosine annealing over 200 epochs |
| Batch | 128 |
| Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip(0.5), per-channel normalization |
| Seeds | 0, 1, 2 |

## Engineering requirements

1. ResNet-18 is implemented from scratch in `src/repro/model.py`. `torchvision.models.resnet18`
   is used **only** inside a test, as a shape/parameter sanity check.
2. Seeding covers python `random`, numpy, torch (CPU + CUDA), and cuDNN (deterministic,
   no benchmark). `tests/test_seed_determinism.py` asserts that two short runs with the same
   seed give bit-identical loss at step 10.
3. A per-step training log (`train_log.csv`) with columns
   `epoch, step, lr, train_loss, train_acc, val_loss, val_acc, wall_time`.
4. Eval (`eval.json`): top-1, top-5, per-class accuracy, and the 10x10 confusion matrix, computed
   on the **final-epoch** model (no best-epoch selection on the test set).
5. `results/aggregate.csv`: one row per seed, plus `mean` and `std` rows.
6. The results table in FINDINGS.md is **generated** from `results/aggregate.csv` by
   `python -m repro.report`. It is never typed by hand, and a test fails if it drifts from the CSV.
7. Plots: `results/accuracy_curve.png`, `results/confusion.png`.

## Acceptance commands (fresh clone)

| Command | Must |
|---|---|
| `make install` | create `.venv`, install torch + package + dev deps |
| `make test` | pass in **< 60 s on CPU** |
| `make docker` | build the Docker image |
| `make smoke` | train 2 epochs on a subset and **assert top-1 > 20%** (sanity only) |
| `make repro` | train 3 seeds × 200 epochs → `results/` → aggregate → plots → regenerate FINDINGS table |

CI (GitHub Actions) runs `make test` and `make smoke` on every push, and builds the Docker image.

## FINDINGS.md must contain

- Citation and link for the paper and for the source of the reference number.
- The reference number.
- A generated table: mean ± std top-1 over 3 seeds, plus per-seed rows.
- The delta and what it means. If we beat the reference, by how much, and a skeptical look
  at why.
- "What diverged and why": at least 3 concrete gaps, each marked resolved or accepted.
- "What I would do with 10x compute".

## README must contain

A one-liner, "What this is / is NOT", reproduction in 3 commands, a link to FINDINGS.md,
hardware, total wall-clock time, total GPU-hours, and known limitations.

## Forbidden

- Using `torchvision.models.resnet18` as the model.
- Typing numbers by hand into the FINDINGS results table.
- Claiming reproduction from one seed.
- Cherry-picking the best seed or the best epoch as the headline number.
