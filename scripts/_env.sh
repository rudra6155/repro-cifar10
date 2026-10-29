# Sourced by the other scripts: repo root as cwd, and $PY pointing at the project interpreter.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
  elif [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
  else PY=python; fi
fi
CONFIG=${CONFIG:-src/repro/configs/resnet18_cifar10.yaml}
export PYTHONUNBUFFERED=1
