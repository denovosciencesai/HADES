import os, warnings, logging, contextlib
from functools import partialmethod

_silenced = False

def silence_libraries():
    global _silenced
    if _silenced:
        return
    _silenced = True
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore")
    for name in ("chemprop", "train", "admet_ai"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from tqdm import tqdm as _t
        _t.__init__ = partialmethod(_t.__init__, disable=True)
    except Exception:
        pass
    try:
        from tqdm.auto import tqdm as _ta
        _ta.__init__ = partialmethod(_ta.__init__, disable=True)
    except Exception:
        pass

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

silence_libraries()