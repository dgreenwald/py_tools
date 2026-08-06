import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


_DOTENV_LOADED = False


def _ensure_dotenv_loaded():
    """Load .env once, if python-dotenv is available.

    Search from current working directory upward for a ``.env`` file.
    Fall back to the repository root (parent of the datasets package) if
    no ``.env`` file is found during the upward search.  Subsequent calls
    are no-ops because the result is cached in the module-level
    ``_DOTENV_LOADED`` flag.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    if load_dotenv is not None:
        cwd = Path.cwd()
        found = False
        for path in (cwd, *cwd.parents):
            candidate = path / ".env"
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)
                found = True
                break

        if not found:
            repo_root = Path(__file__).resolve().parent.parent
            candidate = repo_root / ".env"
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)

    _DOTENV_LOADED = True


def base_dir():
    """Return the base dataset directory as a string ending with a path separator.

    Resolves the data root in priority order:

    1. The ``PY_TOOLS_DATA_DIR`` environment variable (set directly or via
       a ``.env`` file discovered by ``_ensure_dotenv_loaded``).
    2. A legacy fallback of ``~/Dropbox/data`` for backward compatibility.

    Returns
    -------
    str
        Absolute path to the base dataset directory, always ending with
        ``os.sep``.
    """
    return str(base_path()) + os.sep


def base_path():
    """Return the base dataset directory as a :class:`pathlib.Path`.

    Calls ``_ensure_dotenv_loaded`` to pick up any ``.env`` configuration,
    then resolves the data root using the same priority order as
    ``base_dir``: the ``PY_TOOLS_DATA_DIR`` environment variable first,
    then ``~/Dropbox/data`` as a legacy fallback.

    Returns
    -------
    pathlib.Path
        Absolute path to the base dataset directory.
    """
    _ensure_dotenv_loaded()

    env_dir = os.environ.get("PY_TOOLS_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    return Path.home() / "Dropbox" / "data"


def dataset_dir(stub):
    """Return a dataset-specific subdirectory as a string ending with a path separator.

    Appends ``stub`` to the base dataset directory returned by
    ``base_path`` and converts the result to a string.

    Parameters
    ----------
    stub : str
        Subdirectory name (or relative path) beneath the base dataset
        directory.

    Returns
    -------
    str
        Absolute path to the dataset subdirectory, always ending with
        ``os.sep``.
    """
    return str(dataset_path(stub)) + os.sep


def dataset_path(stub):
    """Return a dataset-specific subdirectory as a :class:`pathlib.Path`.

    Appends ``stub`` to the path returned by ``base_path``.

    Parameters
    ----------
    stub : str
        Subdirectory name (or relative path) beneath the base dataset
        directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the dataset subdirectory.
    """
    return base_path() / stub
