"""
HADES_CALCULATOR.py
===================
Score SMILES with the trained HADES ensemble.

As a library:
    from HADES_CALCULATOR import HADES
    hades = HADES()
    hades.score("CCO")                            # -> 0.1234
    hades.score(["CCO", "c1ccccc1"])              # -> DataFrame

From the command line:
    python HADES_CALCULATOR.py input.csv output.csv --smiles-col SMILES

Console noise is suppressed by default; use HADES(quiet=False) to see it.
"""
from HADES import quiet_noise  # must come first: mutes tqdm/chemprop before they load

import os
import glob
import argparse
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
        self.quiet = quiet

        if not os.path.exists(x_train_parquet):
            raise FileNotFoundError(
                f"Could not find '{x_train_parquet}'. Run this from the repo "
                "root, or pass x_train_parquet=<path> to HADES()."
            )
        x_train = pd.read_parquet(x_train_parquet)
        self.descriptor_cols = list(x_train.columns)[:-1]  # last column is SMILES
        # Training-set means, used to fill any NaN produced when featurizing
        # new molecules. Kept from X_train so scoring never depends on the
        # batch being scored.
        self.mean_values = x_train[self.descriptor_cols].mean(axis=0)

        paths = glob.glob(os.path.join(models_dir, "*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No .pkl files found in '{models_dir}'.")
        self.models = {
            os.path.splitext(os.path.basename(p))[0]: joblib.load(p) for p in paths
        }

        with quiet_noise.suppress_output(self.quiet):
            self.featurizer = Featurizer()

    def _featurize(self, smiles_list: list) -> pd.DataFrame:
        with quiet_noise.suppress_output(self.quiet):
            df = self.featurizer.featurize_many_smiles(smiles_list)
        df.columns = df.columns.astype(str)
        df = df[self.descriptor_cols].apply(pd.to_numeric, errors="coerce")
        return df.fillna(self.mean_values)

    def score(self, smiles: Union[str, list]) -> Union[float, pd.DataFrame]:
        """Score one SMILES string (-> float) or a list of them (-> DataFrame)."""
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)

        X = self._featurize(smiles_list)
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        scores = np.mean(probs, axis=0)

        if single:
            return float(scores[0])
        return pd.DataFrame({"SMILES": smiles_list, "HADES_Score": scores})


def main():
    parser = argparse.ArgumentParser(description="Score SMILES in a CSV with HADES")
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("output_csv", help="Path to write output CSV")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    parser.add_argument("--models-dir", default="./HADES_saved_models")
    parser.add_argument("--x-train", default="./train_test_saved/X_train.parquet")
    parser.add_argument("--verbose", action="store_true", help="Show library output")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input_csv)
    hades = HADES(
        models_dir=args.models_dir,
        x_train_parquet=args.x_train,
        quiet=not args.verbose,
    )
    result = hades.score(df_in[args.smiles_col].tolist())

    df_out = df_in.merge(
        result, left_on=args.smiles_col, right_on="SMILES", how="left"
    )
    df_out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df_out)} scored rows to {args.output_csv}")


if __name__ == "__main__":
    main()