# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python toolkit focused on numerical economics and data utilities.
- Core modules live at the repo root (for example `state_space.py`, `mcmc.py`, `time_series.py`, `numerical.py`).
- Dataset loaders and source-specific ingestion code live in `datasets/`.
- Tests and executable validation scripts live in `test/`.
- Legacy experiments have been removed; use git history for historical reference and do not reintroduce them for new features.

Imports assume the package name `py_tools`; set `PYTHONPATH` to the parent of this directory.

## Build, Test, and Development Commands
There is no build step; development is direct Python execution.
- `export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH`: make `py_tools.*` imports resolve.
- `python -m pytest test`: run test discovery (best for `test_*.py` files).
- `python test/test_state_space.py`: run an individual script-style test.
- `python -m compileall .`: quick syntax validation across modules.

If you use dataset loaders, also set `PY_TOOLS_DATA_DIR=/path/to/data`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear module-level function names.
- Prefer `snake_case` for functions/variables and `CapWords` for classes (for example `StateSpaceModel`).
- Keep modules focused by domain (time series, estimation, datasets) rather than adding large multi-purpose files.
- Maintain import style already used in repo (`import py_tools.econ as ec`, `from py_tools.discrete import DiscreteModel`).

## Testing Guidelines
- Add tests under `test/` using `test_<feature>.py` naming.
- Prefer deterministic tests with explicit assertions; avoid relying only on plots/manual inspection.
- For numerical routines, assert tolerances explicitly (for example `np.allclose(..., atol=1e-8)`).
- Run targeted files first, then `python -m pytest test` before opening a PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects (for example `update to crsp`, `adding config registry file`). Keep the first line <= 72 chars.
- Commit one logical change at a time.
- PRs should include: purpose, modules touched, test evidence/commands run, and any data-path assumptions.
- Link related issues when applicable and include before/after notes for behavior changes.

## Security & Configuration Tips
- Do not commit local data paths, credentials, or generated data artifacts.
- Keep environment-specific settings in shell environment variables (`PYTHONPATH`, `PY_TOOLS_DATA_DIR`), not in source files.
