# py_tools Refactoring Plan

## Overview

This document outlines a plan for refactoring `py_tools` to improve its organization, code quality, shareability, and maintainability. The package is a comprehensive econometric and financial research toolkit containing ~90 Python files across core modules and dataset loaders.

The plan is organized into phases that can be executed incrementally without breaking existing usage.

---

## Phase 1: Package Infrastructure

**Goal:** Turn py_tools into a proper installable Python package.

### 1.1 Add `pyproject.toml`

Currently the package has no `setup.py` or `pyproject.toml` and relies on manual `PYTHONPATH` configuration. Adding a `pyproject.toml` enables `pip install -e .` for development and proper dependency declaration.

```
[project]
name = "py_tools"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "statsmodels",
]

[project.optional-dependencies]
ml = ["scikit-learn", "patsy"]
mpi = ["mpi4py"]
scraping = ["beautifulsoup4", "requests"]
nlp = ["nltk"]
all = ["scikit-learn", "patsy", "mpi4py", "beautifulsoup4", "requests", "nltk"]
dev = ["pytest", "pytest-cov", "flake8"]
```

### 1.2 Add proper `__init__.py`

The current `__init__.py` is empty. Add version info and controlled public API exports to make it clear what is part of the package's stable interface.

### 1.3 Add a `pytest.ini` or `pyproject.toml [tool.pytest]` section

Configure test discovery so `pytest` works out of the box.

### 1.4 Update `.gitignore`

Add common entries: `.eggs/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, etc.

### 1.5 Update `README.md`

Replace the current minimal README with:
- Package description and purpose
- Installation instructions (`pip install -e .`)
- Quick-start usage examples
- Module overview / API summary
- Contributing guidelines

---

## Phase 2: Module Reorganization

**Goal:** Group related modules into subpackages to reduce the flat 44-file top-level layout.

### Proposed Subpackage Structure

```
py_tools/
├── __init__.py
├── econometrics/          # Estimation & econometric methods
│   ├── __init__.py
│   ├── estimation.py      # NLS, standard errors
│   ├── gmm.py             # Generalized Method of Moments
│   ├── bootstrap.py       # Bootstrap inference
│   ├── local_projections.py  # LP / LP-IV
│   ├── reghdfe.py         # High-dimensional fixed effects
│   └── machine_learning.py  # ML utilities
│
├── bayesian/              # Bayesian inference tools
│   ├── __init__.py
│   ├── mcmc.py            # MCMC sampling
│   └── prior.py           # Prior distributions
│
├── time_series/           # Time series models
│   ├── __init__.py
│   ├── core.py            # Core TS utilities (from time_series.py)
│   ├── var.py             # VAR models (from vector_autoregression.py)
│   ├── bvar.py            # Bayesian VAR
│   ├── state_space.py     # State space models (consolidated)
│   ├── kalman.py          # Kalman filter
│   └── hidden_markov.py   # Hidden Markov models
│
├── stats/                 # Statistical tools
│   ├── __init__.py
│   ├── core.py            # Statistical functions (from stats.py)
│   ├── inequality.py      # Inequality measures
│   └── walker.py          # Walker alias sampling
│
├── numerical/             # Numerical methods
│   ├── __init__.py
│   ├── core.py            # Numerical routines (from numerical.py)
│   ├── newton.py          # Newton solver
│   └── chebyshev.py       # Chebyshev approximation
│
├── compute/               # Parallel & high-performance computing
│   ├── __init__.py
│   └── mpi_array.py       # MPI parallelization
│
├── data/                  # Data handling & I/O
│   ├── __init__.py
│   ├── core.py            # Data manipulation (from data.py)
│   ├── collapser.py       # Data collapsing
│   └── match.py           # Matching algorithms
│
├── in_out/                # I/O and external format interfaces
│   ├── __init__.py
│   ├── core.py            # I/O utilities (from in_out.py; renamed from io.py)
│   └── stata.py           # Stata interface
│
├── econ/               # Financial & economic tools
│   ├── __init__.py
│   ├── aim.py             # AIM solver
│   ├── discrete.py        # Discrete choice models
│   ├── financial.py       # Yield curves, coupon math
│   └── econ.py            # Economic utilities
│
├── text/                  # Text and code-formatting utilities
│   ├── __init__.py
│   ├── core.py            # Text formatting (from text.py)
│   ├── parsing.py         # Parsing helpers (from parsing.py)
│   └── format_code.py     # Code formatting
│
├── plot/                  # Plotting and output visuals
│   ├── __init__.py
│   └── core.py            # Matplotlib wrappers (from plot.py)
│
├── utilities/             # Shared utility helpers and containers
│   ├── __init__.py
│   ├── core.py            # General-purpose utilities (from utilities.py)
│   └── containers.py      # Custom data structures
│
├── scraping/              # Web data acquisition
│   ├── __init__.py
│   └── scrape.py          # General web scraping
│
├── config/                # Configuration
│   ├── __init__.py
│   ├── registry.py        # Config registry (from config_registry.py)
│   └── modspec.py         # Model specification
│
├── datasets/              # Dataset loaders (keep as-is, see Phase 3)
│   ├── __init__.py
│   ├── loader.py
│   └── ... (existing dataset modules)
│
└── tests/                 # Test suite (see Phase 5)
    └── ...
```

### Migration Strategy

To avoid breaking existing imports immediately, use **re-exports** from the old locations:

1. Move each module into its new subpackage.
2. In the top-level `py_tools/` namespace, add compatibility imports that re-export from the new location with a deprecation warning.
3. After a transition period, remove the compatibility shims.

Example compatibility shim (in top-level `py_tools/gmm.py`):
```python
import warnings
warnings.warn(
    "Importing from py_tools.gmm is deprecated. "
    "Use py_tools.econometrics.gmm instead.",
    DeprecationWarning, stacklevel=2
)
from py_tools.econometrics.gmm import *
```

---

## Phase 3: Consolidate Duplicates and Clean Up

### 3.1 Merge `state_space.py` and `state_space_new.py`

These two files both implement `StateSpaceModel` with the newer version adding time-varying matrix support. Consolidate into a single `state_space.py` in the `time_series/` subpackage, keeping the enhanced functionality from `state_space_new.py`.

### 3.2 Remove or archive `old/` directory

The `old/hidden_markov_old.py` should be removed from the repository. If historical reference is needed, it is preserved in git history.

### 3.3 Audit `learn.py` and `machine_learning.py`

These have overlapping concerns. Evaluate whether they should be merged or if `learn.py` serves a distinct purpose (e.g., online learning vs. batch ML).

### 3.4 Evaluate `flatten.py`

`flatten.py` handles LaTeX command expansion and depends on `parsing.py`. Consider whether this belongs in a `visualization/` or `text/` subpackage, or if it is specialized enough to remain standalone.

---

## Phase 4: Improve the `datasets/` Subpackage

### 4.1 Standardize dataset loader interface

Each dataset module should follow a consistent pattern:

```python
# Every dataset module should implement:
def load(data_dir=None, **kwargs) -> pd.DataFrame:
    """Load and return the dataset as a DataFrame."""
    ...

# Optional:
def download(data_dir=None, **kwargs):
    """Download raw data files."""
    ...

DATASET_NAME = "descriptive_name"
DESCRIPTION = "One-line description of the dataset."
```

Audit all 38 dataset modules for compliance and add missing docstrings.

### 4.2 Improve `datasets/__init__.py`

Replace the current minimal init with a registry of available datasets:

```python
AVAILABLE_DATASETS = {
    "fred": "Federal Reserve Economic Data (FRED)",
    "crsp": "CRSP Stock Database",
    "compustat": "Compustat Financial Data",
    # ...
}

def list_datasets():
    """List all available dataset loaders."""
    return AVAILABLE_DATASETS
```

### 4.3 Group dataset modules into subdirectories (optional)

If the 38-file flat structure becomes unwieldy, consider grouping:

```
datasets/
├── macro/       # fred, nipa, bea_industry, spf, jst, romer
├── finance/     # crsp, compustat, fama_bliss, shiller, french
├── housing/     # zillow, fhfa, gse, hmda, ahs, fannie
├── admin/       # irs, census, cfpb, saez
└── utils/       # loader, defaults, crosswalk, ds_comp_merge
```

This is lower priority since the current flat structure works and the dataset modules are relatively independent.

---

## Phase 5: Testing

### 5.1 Establish test infrastructure

- Move `test/` to `tests/` (standard convention).
- Add `conftest.py` with shared fixtures.
- Configure pytest in `pyproject.toml`.

### 5.2 Expand test coverage

Current coverage is minimal (only 3 test files for ~44 core modules). Prioritize tests for:

| Priority | Modules | Reason |
|----------|---------|--------|
| High | `numerical.py`, `stats.py`, `data.py` | Core utilities used everywhere |
| High | `estimation.py`, `gmm.py` | Core estimation — correctness is critical |
| Medium | `time_series.py`, `kalman.py` | Widely used TS functions |
| Medium | `state_space.py`, `hidden_markov.py` | Complex algorithms |
| Medium | `financial.py`, `econ.py` | Domain-specific calculations |
| Lower | `plot.py`, `text.py` | Output/display (less critical) |
| Lower | `datasets/*` | Data loading (harder to test without data) |

### 5.3 Add CI via GitHub Actions

Create a basic `.github/workflows/test.yml`:
- Run `pytest` on push and PR
- Test on Python 3.8+
- Lint with `flake8` or `ruff`

---

## Phase 6: Code Quality Improvements

### 6.1 Add docstrings

Many functions and classes lack docstrings. Adopt NumPy-style docstrings consistently:

```python
def estimate_nls(y, X, theta0, model_func, **kwargs):
    """Estimate parameters via nonlinear least squares.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable, shape (n,).
    X : np.ndarray
        Independent variables, shape (n, k).
    theta0 : np.ndarray
        Initial parameter guess, shape (p,).
    model_func : callable
        Model function f(X, theta) -> y_hat.

    Returns
    -------
    results : dict
        Dictionary with keys 'theta', 'se', 'residuals'.
    """
```

### 6.2 Add type hints

Gradually add type annotations, starting with the most-used utility modules (`utilities.py`, `numerical.py`, `stats.py`, `data.py`).

### 6.3 Lint and format

- Run `ruff` or `flake8` to identify issues.
- Apply `black` or `ruff format` for consistent formatting.
- Address unused imports and variables.

### 6.4 Review and clean up `utilities.py`

`utilities.py` is a catch-all module. Audit its contents and move functions to more specific modules where appropriate (e.g., timing functions to a `timing` module, list helpers to `data/core.py`).

---

## Phase 7: Dependency Management

### 7.1 Lazy imports for heavy/optional dependencies

Some modules import heavyweight or uncommon packages (`mpi4py`, `pyopencl`, `nltk`, `beautifulsoup4`). Use lazy imports so the base package can be installed without these:

```python
def run_mpi_task(...):
    from mpi4py import MPI
    # ...
```

### 7.2 Pin minimum versions

In `pyproject.toml`, specify minimum compatible versions for core dependencies based on the API features actually used.

---

## Execution Order and Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Package Infrastructure | 🔴 High | Low | High — enables `pip install`, proper deps |
| 2. Module Reorganization | 🟡 Medium | High | High — improves discoverability |
| 3. Consolidate Duplicates | 🔴 High | Low | Medium — reduces confusion |
| 5.1 Test Infrastructure | 🔴 High | Low | High — enables safe refactoring |
| 6.1 Docstrings | 🟡 Medium | Medium | Medium — improves usability |
| 4. Dataset Improvements | 🟡 Medium | Medium | Medium — consistency |
| 5.2 Expand Tests | 🟡 Medium | High | High — catches regressions |
| 6.2–6.4 Type Hints, Lint | 🟢 Low | Medium | Medium — long-term quality |
| 7. Dependency Management | 🟢 Low | Low | Medium — shareability |
| 5.3 CI | 🟢 Low | Low | High — automation |

### Recommended first steps

1. **Phase 1** — Add `pyproject.toml` and make the package installable.
2. **Phase 3.1** — Consolidate `state_space.py` / `state_space_new.py`.
3. **Phase 3.2** — Remove the `old/` directory.
4. **Phase 5.1** — Set up pytest properly.
5. **Phase 2** — Begin module reorganization (one subpackage at a time), with compatibility shims.

---

## Risks and Considerations

- **Breaking existing projects:** The compatibility shim approach (Phase 2) mitigates this but requires discipline. All existing `from py_tools.X import Y` statements in downstream projects will need eventual updates.
- **Dataset data paths:** The `PY_TOOLS_DATA_DIR` environment variable pattern should be preserved. Consider also supporting a config file (e.g., `~/.py_tools.toml`).
- **MPI dependencies:** `mpi4py` requires a system MPI installation. Ensure this remains optional and doesn't block basic package usage.
- **Scope creep:** This plan is intentionally focused on organization and quality, not on adding new features or rewriting algorithms.
