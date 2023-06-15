def autoreload() -> None:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

def add_dirs(rel_dirs: list) -> None:
    import sys, pathlib
    """Add a list of directories to PATH"""
    for r in rel_dirs:
        a = pathlib.Path(r).expanduser().resolve(strict=True)
        if (str(a) not in sys.path) and (a.exists()):
            sys.path.insert(0, str(a))