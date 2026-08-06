# PyTools

`py_tools` is a research-oriented Python toolkit for numerical economics, time series, estimation, and dataset loading utilities.

## Installation

Install from PyPI:

```bash
pip install dgreenwald-py-tools
```

Or clone the repository and install in editable mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install "dgreenwald-py-tools[ml]"        # scikit-learn, patsy
pip install "dgreenwald-py-tools[datasets]"  # pandas-datareader, python-dotenv
pip install "dgreenwald-py-tools[scraping]"  # requests, beautifulsoup4, lxml
pip install "dgreenwald-py-tools[nlp]"       # nltk
pip install "dgreenwald-py-tools[mpi]"       # mpi4py
pip install "dgreenwald-py-tools[dev]"       # pytest, ruff
```

If you use dataset loaders, set:

```bash
export PY_TOOLS_DATA_DIR=/path/to/data
```

Or create a `.env` file in the repository root (or a parent directory):

```env
PY_TOOLS_DATA_DIR=/path/to/data
```

`py_tools.datasets` will load `.env` automatically when `python-dotenv` is installed
(included in the `datasets` extra).

## Quick Start

```python
from py_tools import data, time_series, state_space

# Example: call a utility function from a core module
# (see module docstrings/source for full APIs)
```

You can also import dataset loaders through:

```python
from py_tools import datasets

available = datasets.list_datasets()
df = datasets.load_dataset("fred", codes=["UNRATE"])
```

## Module Overview

| Import path | Contents |
|---|---|
| `py_tools.time_series` | Kalman filter, state-space models, VAR/BVAR, HMMs |
| `py_tools.econometrics` | NLS/GMM, bootstrap, local projections, high-dimensional FE |
| `py_tools.bayesian` | MCMC sampling, prior distributions |
| `py_tools.numerical` | Root-finding, Chebyshev approximation |
| `py_tools.datasets` | Loaders for ~38 economic data sources |
| `py_tools.data` | Data manipulation, aggregation, matching |
| `py_tools.econ` | Discrete choice, yield curves, AIM solver |
| `py_tools.plot` | Plotting utilities |
| `py_tools.config` | Named model specification registry |
| `py_tools.scraping` | HTML scraping and text utilities |
| `py_tools.compute` | MPI array distribution |

## Development

```bash
pip install -e ".[dev]"
python -m pytest          # run tests
ruff check .              # lint
```

## Contributing

- Keep changes focused and commit one logical change at a time.
- Follow repository coding and workflow guidelines in `AGENTS.md`.
- Prefer adding deterministic tests in `tests/` for new behavior.

## License

MIT — see [LICENSE](LICENSE).
