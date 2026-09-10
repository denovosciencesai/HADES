from HADES.quiet_noise import suppress_output
from HADES.featurize import Featurizer

import argparse, contextlib, glob, os
from typing import Union
import joblib, numpy as np, pandas as pd


class HADES_score:
    """
    Loads trained HADES models and scores molecules (given as SMILES strings)
    with an ensemble-averaged probability score.

    On init, reads the training feature matrix to recover the descriptor column
    order and per-column means (used later for NaN imputation), loads every
    `.pkl` model found in `models_dir` into an ensemble, and builds a
    `Featurizer` matching the training descriptor columns.
    """
    def __init__(self, models_dir="./HADES_saved_models",
                 x_train_parquet="./train_test_saved/X_train.parquet", quiet=True, fillna="mean"):
        """
        Args:
            models_dir (str): Directory containing trained model `.pkl` files.
                Every file found here is loaded and included in the scoring ensemble,
                keyed by filename (without extension).
            x_train_parquet (str): Path to the training feature parquet. Its columns
                (excluding the last, assumed to be the target) define the expected
                descriptor order, and their means are used for mean-imputation.
            quiet (bool): If True, suppresses stdout/stderr noise from the featurizer
                during construction and scoring. Defaults to True.
            fillna (str or scalar or None): Strategy for filling missing/invalid
                descriptor values after featurization. "mean" fills with the
                corresponding training-set column mean; any other non-None value is
                used as a literal fill value; None leaves NaNs as-is. Defaults to "mean".
        """
        self.quiet = quiet
        self.fillna = fillna
        x_train = pd.read_parquet(x_train_parquet)
        self.descriptor_cols = list(x_train.columns)[:-1]
        self._train_means = x_train[self.descriptor_cols].mean()
        self.models = {os.path.splitext(os.path.basename(p))[0]: joblib.load(p)
                       for p in glob.glob(os.path.join(models_dir, "*.pkl"))}
        with self._q():
            self.featurizer = Featurizer.from_columns(self.descriptor_cols)
        self._feature_names = self.featurizer.feature_names

    def _q(self):
        """
        Returns the appropriate context manager for the instance's quiet setting:
        `suppress_output()` if `self.quiet` is True, otherwise a no-op
        `contextlib.nullcontext()`.
        """
        return suppress_output() if self.quiet else contextlib.nullcontext()

    def _featurize(self, smiles_list):
        """
        Converts a list of SMILES strings into a numeric feature DataFrame aligned
        to the training descriptor columns.
        Args:
            smiles_list (list[str]): SMILES strings to featurize.
        Returns:
            pd.DataFrame: Feature matrix with columns matching `self.descriptor_cols`.
                Unparseable SMILES get no row; the index gives their position in
                `smiles_list`.
        """
        with self._q():
            if len(smiles_list) == 1:
                feats = self.featurizer.featurize_smiles(smiles_list[0])
                df = pd.DataFrame([feats] if feats else [], columns=self._feature_names)
            else:
                df = self.featurizer.featurize_many_smiles(smiles_list)
        df.columns = df.columns.astype(str)
        df = df.reindex(columns=self.descriptor_cols).apply(pd.to_numeric, errors="coerce")
        if self.fillna == "mean":
            df = df.fillna(self._train_means)
        elif self.fillna is not None:
            df = df.fillna(self.fillna)
        return df

    def score(self, smiles: Union[str, list[str]]):
        """
        Scores one or more molecules by averaging the positive-class probability
        across all loaded models.
        Args:
            smiles (str or list[str]): A single SMILES string, or a list of SMILES
                strings.
        Returns:
            float: The averaged HADES score, if `smiles` is a single string.
            pd.DataFrame: A DataFrame with "SMILES" and "HADES_Score" columns,
                if `smiles` is a list. Unparseable SMILES are skipped and listed
                on stdout.
        Raises:
            ValueError: If `smiles` is a single string that RDKit cannot parse.
        """
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        with self._q():
            X = self._featurize(smiles_list)
            scores = (np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
                      if len(X) else np.array([]))
        kept = list(X.index)
        bad = [s for i, s in enumerate(smiles_list) if i not in set(kept)]
        if single:
            if bad:
                raise ValueError(f"Not a valid SMILES: {smiles!r}")
            return float(scores[0])
        if bad:
            print(f"Could not calculate {len(bad)} SMILES: {bad}")
        return pd.DataFrame({"SMILES": [smiles_list[i] for i in kept], "HADES_Score": scores})

    def score_csv(self, input_csv, output_csv, smiles_col="SMILES"):
        """
        Scores all molecules in a CSV file and writes the result to a new CSV.
        Args:
            input_csv (str): Path to the input CSV containing a SMILES column.
            output_csv (str): Path to write the scored output CSV.
            smiles_col (str): Name of the column containing SMILES strings.
                Defaults to "SMILES".
        Returns:
            pd.DataFrame: The merged input + score DataFrame that was written to disk.
        """
        df_in = pd.read_csv(input_csv)
        df_out = df_in.merge(self.score(df_in[smiles_col].tolist()), left_on=smiles_col, right_on="SMILES", how="left")
        df_out.to_csv(output_csv, index=False)
        print(f"Wrote {len(df_out)} scored rows to {output_csv}")
        return df_out


def main():
    """
    Command-line entry point for HADES_score.

    Parses arguments for model/training-data locations and an optional input/output
    CSV pair, then either:
      - scores the SMILES column of `--input` and writes results to `--output`
        (both must be provided together), or
      - if neither is provided, runs a small built-in demo, printing scores for
        four example SMILES strings.

    Raises an argparse error if only one of `--input`/`--output` is given.
    """
    p = argparse.ArgumentParser(description="Score SMILES with HADES. No args = demo.")
    p.add_argument("-i", "--input", dest="input_csv", help="Path to input CSV")
    p.add_argument("-o", "--output", dest="output_csv", help="Path to write output CSV")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--models-dir", default="./HADES_saved_models",
                   help="Directory containing trained model .pkl files")
    p.add_argument("--x-train", dest="x_train_parquet", default="./train_test_saved/X_train.parquet",
                   help="Path to X_train.parquet used to derive descriptor columns and fill-na means")
    args = p.parse_args()

    hades = HADES_score(models_dir=args.models_dir, x_train_parquet=args.x_train_parquet)

    if args.input_csv and args.output_csv:
        hades.score_csv(args.input_csv, args.output_csv, smiles_col=args.smiles_col)
    elif args.input_csv or args.output_csv:
        p.error("both -i/--input and -o/--output are required together")
    else:
        print(hades.score(["NC(=O)N1CCN(C(=O)Cc2ccc(CN3CCc4ccccc43)cc2)CC1",
                            "Cc1cccnc1NC(=O)c1cccc(C(=O)N2CCN(C)CC2)c1", "C1CCCC1", "CCO"]))


if __name__ == "__main__":
    main()