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
    base_path = Path(__file__).resolve().parent
    for path in sorted(base_path.glob("*.py")):
        module_name = path.stem
        if module_name in _UTILITY_MODULES:
            continue
        yield module_name, path


def _extract_metadata(path, default_name):
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
    available = {}

    for module_name, path in _iter_dataset_module_paths():
        dataset_name, description = _extract_metadata(path, module_name)
        available[dataset_name] = description
        _DATASET_MODULES[dataset_name] = module_name

    return {key: available[key] for key in sorted(available)}


AVAILABLE_DATASETS = _build_registry()


def list_datasets():
    """Return a copy of available dataset loaders and descriptions."""
    return AVAILABLE_DATASETS.copy()


def load_dataset(name, **kwargs):
    """Load a dataset by registry name using the module-level `load` function."""
    if name not in AVAILABLE_DATASETS:
        available = ", ".join(sorted(AVAILABLE_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

    module_name = _DATASET_MODULES[name]
    module = import_module(f"{__name__}.{module_name}")
    if not hasattr(module, "load"):
        raise AttributeError(f"Dataset module '{module_name}' does not define load()")
    return module.load(**kwargs)
