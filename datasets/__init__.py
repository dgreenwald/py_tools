import ast
from importlib import import_module
from pathlib import Path

_UTILITY_MODULES = {
    "__init__",
    "config",
    "crosswalk",
    "download_ahs",
    "ds_comp_merge",
    "frm_server",
    "loader",
}

_DATASET_MODULES = {}


def _iter_dataset_module_paths():
    """Yield (module_name, path) tuples for every dataset module in this package.

    Scans the package directory for ``*.py`` files and skips any whose stem
    appears in ``_UTILITY_MODULES`` (e.g. ``__init__``, ``config``,
    ``crosswalk``).

    Yields
    ------
    module_name : str
        Stem of the Python file (e.g. ``"ahs"``).
    path : pathlib.Path
        Absolute path to the Python source file.
    """
    base_path = Path(__file__).resolve().parent
    for path in sorted(base_path.glob("*.py")):
        module_name = path.stem
        if module_name in _UTILITY_MODULES:
            continue
        yield module_name, path


def _extract_metadata(path, default_name):
    """Parse a dataset module source file and extract its registry metadata.

    Uses :mod:`ast` to statically read the module-level ``DATASET_NAME`` and
    ``DESCRIPTION`` string constants without importing the module.  Falls back
    to ``default_name`` / a generic description string when the file cannot be
    parsed or the constants are absent.

    Parameters
    ----------
    path : pathlib.Path
        Absolute path to the Python source file to inspect.
    default_name : str
        Fallback registry key to use when ``DATASET_NAME`` is not defined in
        the source file.

    Returns
    -------
    dataset_name : str
        Value of ``DATASET_NAME`` found in the source, or ``default_name`` if
        not present.
    description : str
        Value of ``DESCRIPTION`` found in the source, or a generic placeholder
        derived from ``default_name`` if not present.
    """
    dataset_name = default_name
    description = f"{default_name} dataset loader."

    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return dataset_name, description

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "DATASET_NAME" and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    dataset_name = node.value.value
            elif target.id == "DESCRIPTION" and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    description = node.value.value

    return dataset_name, description


def _build_registry():
    """Scan all dataset modules and build the name-to-description registry.

    Iterates every path returned by :func:`_iter_dataset_module_paths`, calls
    :func:`_extract_metadata` for each, and populates the module-level
    ``_DATASET_MODULES`` mapping (dataset name → Python module stem) as a
    side-effect.

    Returns
    -------
    dict
        A sorted ``{dataset_name: description}`` mapping for all discovered
        dataset modules.
    """
    available = {}

    for module_name, path in _iter_dataset_module_paths():
        dataset_name, description = _extract_metadata(path, module_name)
        available[dataset_name] = description
        _DATASET_MODULES[dataset_name] = module_name

    return {key: available[key] for key in sorted(available)}


AVAILABLE_DATASETS = _build_registry()


def list_datasets():
    """Return a copy of available dataset loaders and descriptions.

    Provides a snapshot of the registry built at import time.  The returned
    dict is a shallow copy, so mutations do not affect the module-level
    ``AVAILABLE_DATASETS`` registry.

    Returns
    -------
    dict
        A ``{dataset_name: description}`` mapping, sorted alphabetically by
        dataset name, for every dataset module discovered in this package.
    """
    return AVAILABLE_DATASETS.copy()


def load_dataset(name, **kwargs):
    """Load a dataset by registry name using the module-level ``load`` function.

    Looks up ``name`` in the registry built by :func:`_build_registry`,
    imports the corresponding dataset module, and delegates to its
    ``load(**kwargs)`` function.

    Parameters
    ----------
    name : str
        Registry key for the desired dataset (e.g. ``"ahs"``, ``"census"``).
        Use :func:`list_datasets` to see all valid keys.
    **kwargs
        Arbitrary keyword arguments forwarded verbatim to the dataset module's
        ``load()`` function.

    Returns
    -------
    object
        Whatever the dataset module's ``load()`` function returns (typically a
        :class:`pandas.DataFrame`).

    Raises
    ------
    ValueError
        If ``name`` is not present in the dataset registry.
    AttributeError
        If the resolved dataset module does not define a ``load()`` function.
    """
    if name not in AVAILABLE_DATASETS:
        available = ", ".join(sorted(AVAILABLE_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

    module_name = _DATASET_MODULES[name]
    module = import_module(f"{__name__}.{module_name}")
    if not hasattr(module, "load"):
        raise AttributeError(f"Dataset module '{module_name}' does not define load()")
    return module.load(**kwargs)
