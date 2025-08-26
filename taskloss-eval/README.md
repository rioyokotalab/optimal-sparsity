# taskloss-eval

## environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## usage

```bash
python scripts/run_taskloss.py \
    --model <MODEL> \
    --task <DATASET> \
    --tp 1
```
