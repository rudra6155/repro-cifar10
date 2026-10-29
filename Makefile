# repro-cifar10 — run from bash (Linux/macOS, or Git Bash on Windows).
SHELL := bash

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
  # PyPI only ships CPU torch wheels for Windows, so point at a CUDA index by default.
  TORCH_INDEX ?= https://download.pytorch.org/whl/cu126
else
  VENV_PY := .venv/bin/python
  # Linux PyPI wheels already include CUDA. CI overrides this with the CPU index.
  TORCH_INDEX ?=
endif

PYTHON ?= python
PY     ?= $(VENV_PY)
CONFIG ?= src/repro/configs/resnet18_cifar10.yaml
IMAGE  ?= repro-cifar10
SEEDS  ?= 0 1 2

export PY CONFIG SEEDS

.PHONY: help install test smoke repro report check-findings docker docker-test clean

help:
	@echo "make install   create .venv and install torch + package"
	@echo "make test      unit tests (CPU, < 60 s)"
	@echo "make smoke     2 epochs on a subset, asserts top-1 > 20%"
	@echo "make repro     3 seeds x 200 epochs -> results/ -> FINDINGS table"
	@echo "make report    re-aggregate results/ and regenerate plots + FINDINGS table"
	@echo "make docker    build the Docker image ($(IMAGE))"

install:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision $(if $(TORCH_INDEX),--index-url $(TORCH_INDEX))
	$(PY) -m pip install -e ".[dev]"

test:
	CUDA_VISIBLE_DEVICES="" $(PY) -m pytest tests

smoke:
	$(PY) -m repro.train --config $(CONFIG) --smoke --seed 0 --out results/smoke

repro:
	bash scripts/repro.sh

report:
	$(PY) -m repro.report --results results --findings FINDINGS.md --readme README.md

check-findings:
	$(PY) -m repro.report --results results --findings FINDINGS.md --readme README.md --check

docker:
	docker build -t $(IMAGE) .

# Runs the CI checks (test + smoke) inside the Linux CPU image.
docker-test: docker
	docker run --rm $(IMAGE) make test smoke

clean:
	rm -rf results/smoke .pytest_cache
