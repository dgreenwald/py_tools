import os

def base_dir():
    return os.environ.get('PY_TOOLS_DATA_DIR', None)
