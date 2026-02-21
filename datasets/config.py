import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


_DOTENV_LOADED = False


def _ensure_dotenv_loaded():
    """Load .env once, if python-dotenv is available.

    Search from current working directory upward. Fallback to repository root
    (parent of the datasets package) if not found during upward search.
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


def base_dir(user="DAN"):
    """Return base dataset directory as a string ending with a path separator.

    Priority:
    1) Explicit environment variable PY_TOOLS_DATA_DIR
    2) Legacy user-specific HOME/Dropbox fallback
    """
    return str(base_path(user=user)) + os.sep


def base_path(user="DAN"):
    """Return base dataset directory as a pathlib.Path."""
    _ensure_dotenv_loaded()

    env_dir = os.environ.get("PY_TOOLS_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    if user == "DAN":
        return Path.home() / "Dropbox" / "data"
    if user == "MARY":
        return Path.home() / "Dropbox (MIT)" / "data"

    return Path.home() / "Dropbox" / "data"


def dataset_dir(stub, user="DAN"):
    """Return a dataset-specific directory as a string ending with separator."""
    return str(dataset_path(stub, user=user)) + os.sep


def dataset_path(stub, user="DAN"):
    """Return a dataset-specific directory path for a subfolder stub."""
    return base_path(user=user) / stub
