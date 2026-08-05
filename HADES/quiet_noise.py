import os
import warnings
import logging
import contextlib
from functools import partialmethod

# 1) Ask tqdm to disable all progress bars via its environment switch.
os.environ.setdefault("TQDM_DISABLE", "1")

# 2) Hide Python warnings (e.g. the LightGBM min_data_in_leaf notice).
warnings.filterwarnings("ignore")

# 3) Quiet the loggers chemprop / admet_ai use for their "Loading pretrained
#    parameter ..." chatter.
for _name in ("chemprop", "train", "admet_ai"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# 4) Belt-and-braces: force every tqdm bar to start disabled, in case a library
#    creates one directly and ignores the env var above.
try:
    from tqdm import tqdm as _tqdm
    _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
except Exception:
    pass

try:
    from tqdm.auto import tqdm as _tqdm_auto
    _tqdm_auto.__init__ = partialmethod(_tqdm_auto.__init__, disable=True)
except Exception:
    pass

@contextlib.contextmanager
def suppress_output():
    """Silence anything printed straight to stdout/stderr inside this block.
    This is the catch-all for messages that bypass logging/tqdm entirely
    (some chemprop versions print parameter names directly to stdout).
    A genuine exception still propagates and stops execution -- only the
    printed text is discarded.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield