import os

def base_dir(user='DAN'):
    if user=='DAN':
        return os.environ.get('PY_TOOLS_DATA_DIR', os.environ['HOME'] + '/Dropbox/data/')
    elif user=='MARY':
        return os.environ.get('PY_TOOLS_DATA_DIR', os.environ['HOME'] + '/Dropbox (MIT)/data/')