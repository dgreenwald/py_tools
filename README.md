# PyTools

`py_tools` is a research-oriented Python toolkit for numerical economics, time series, estimation, and dataset loading utilities.

## Installation

Clone the repository and install in editable mode:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev]"       # pytest, flake8, coverage tools
python -m pip install -e ".[ml]"        # scikit-learn, patsy
python -m pip install -e ".[scraping]"  # requests, bs4, pandas-datareader
```

If you use dataset loaders, set:

```bash
export PY_TOOLS_DATA_DIR=/path/to/data
```

## Quick Start

```python
from py_tools import data, time_series, state_space

# Example: call a utility function from a core module
# (see module docstrings/source for full APIs)
```

You can also import dataset loaders through:

```python
from py_tools import datasets
```

## Module Overview

- Core modules live at the repository root (for example `time_series.py`, `state_space.py`, `numerical.py`, `data.py`).
- Dataset-specific loaders live in `datasets/`.
- Script-style usage examples live in `examples/`.
- Future unit tests should live in `tests/`.

The top-level package exposes a curated API via lazy imports in `py_tools.__all__`.

## Development

Run test discovery (currently targets `tests/`):

```bash
python -m pytest
```

Run one of the script-style examples directly:

```bash
python examples/test_state_space.py
```

## Contributing

- Keep changes focused and commit one logical change at a time.
- Follow repository coding and workflow guidelines in `AGENTS.md`.
- Prefer adding deterministic tests in `tests/` for new behavior.
