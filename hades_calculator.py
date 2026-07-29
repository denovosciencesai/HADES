
"""
hades_calculator.py
===================
A thin, importable wrapper around the HADES pipeline so you can do:

    from hades_calculator import HADES
    hades = HADES()
    hades.score("CCO")                                 # -> 0.1234  (single SMILES)
    hades.score(["CCO", "CC(=O)Oc1ccccc1C(=O)O"])      # -> DataFrame of scores

It reuses the existing `HADES/featurize.py` Featurizer and the model files
already saved in `HADES_saved_models/`, so nothing is downloaded at run time.

Place this file in the root of the HADES repo and run it from that directory,
or pass explicit paths to HADES(...).

Console noise (tqdm progress bars, chemprop's "Loading pretrained parameter"
lines, and LightGBM warnings) is suppressed by default. To see it again while
debugging, construct with:  HADES(quiet=False).
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Noise suppression. Must run BEFORE the heavy libraries are imported so their #
# module-level tqdm/logging setup is already muted.                            #
# --------------------------------------------------------------------------- #
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
def _suppress_output():
    """Silence anything printed straight to stdout/stderr inside this block.

    This is the catch-all for messages that bypass logging/tqdm entirely
    (some chemprop versions print parameter names directly to stdout).
    A genuine exception still propagates and stops execution -- only the
    printed text is discarded.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# --------------------------------------------------------------------------- #
# Heavy imports (after the suppression setup above).                           #
# --------------------------------------------------------------------------- #
import glob
from typing import Union

import joblib
import numpy as np
import pandas as pd

from HADES.featurize import Featurizer


class HADES:
    """Loads the trained ensemble once, then scores SMILES on demand."""

    def __init__(
        self,
        models_dir: str = "./HADES_saved_models",
        x_train_parquet: str = "./train_test_saved/X_train.parquet",
        quiet: bool = True,
    ):
        # When quiet=True, build the featurizer (which may load chemprop /
        # admet_ai models and print) inside the output suppressor.
        self.quiet = quiet
        self.models_dir = models_dir
        self.descriptor_cols = self._load_descriptor_cols(x_train_parquet)
        self.models = self._load_models(models_dir)
        with self._maybe_quiet():
            self.featurizer = Featurizer()

    # ---- helpers ------------------------------------------------------------
    def _maybe_quiet(self):
        """Return the output suppressor if quiet, else a no-op context."""
        return _suppress_output() if self.quiet else contextlib.nullcontext()

    @staticmethod
    def _load_descriptor_cols(x_train_parquet: str) -> list[str]:
        if not os.path.exists(x_train_parquet):
            raise FileNotFoundError(
                f"Could not find '{x_train_parquet}'. Run this from the repo "
                "root, or pass x_train_parquet=<path> to HADES()."
            )
        x_train = pd.read_parquet(x_train_parquet)
        return list(x_train.columns)[:-1]  # last column is the label (SMILES)

    @staticmethod
    def _load_models(models_dir: str) -> dict:
        paths = glob.glob(os.path.join(models_dir, "*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No .pkl files found in '{models_dir}'.")
        models = {}
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            models[name] = joblib.load(path)
        return models

    def _featurize(self, smiles_list: list[str]) -> pd.DataFrame:
        with self._maybe_quiet():
            df = self.featurizer.featurize_many_smiles(smiles_list)
        df.columns = df.columns.astype(str)
        df[self.descriptor_cols] = df[self.descriptor_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        return df[self.descriptor_cols]

    def score(self, smiles: Union[str, list[str]]) -> Union[float, pd.DataFrame]:
        """
        Compute the HADES score for one SMILES string or a list of them.

        Returns:
            float          if `smiles` is a single string
            pd.DataFrame   if `smiles` is a list (columns: SMILES, HADES_Score)
        """
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        with self._maybe_quiet():
            X = self._featurize(smiles_list)
            probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        scores = np.mean(probs, axis=0)
        if single:
            return float(scores[0])
        return pd.DataFrame({"SMILES": smiles_list, "HADES_Score": scores})


if __name__ == "__main__":
    hades = HADES()
    example_smiles = [
        "NC(=O)N1CCN(C(=O)Cc2ccc(CN3CCc4ccccc43)cc2)CC1",
        "Cc1cccnc1NC(=O)c1cccc(C(=O)N2CCN(C)CC2)c1",
        "C1CCCC1",
        "CCO",
    ]
    print(hades.score(example_smiles))