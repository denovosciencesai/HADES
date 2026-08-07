import argparse, contextlib, glob, os
from typing import Union
import joblib, numpy as np, pandas as pd
from HADES.quiet_noise import suppress_output
from HADES.featurize import Featurizer


class HADES_score:
    def __init__(self, models_dir="./HADES_saved_models",
                 x_train_parquet="./train_test_saved/X_train.parquet", quiet=True, fillna="mean"):
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
        return suppress_output() if self.quiet else contextlib.nullcontext()

    def _featurize(self, smiles_list):
        with self._q():
            if len(smiles_list) == 1:
                feats = self.featurizer.featurize_smiles(smiles_list[0]) or [float("nan")] * len(self._feature_names)
                df = pd.DataFrame([feats], columns=self._feature_names)
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
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        with self._q():
            X = self._featurize(smiles_list)
            scores = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        return float(scores[0]) if single else pd.DataFrame({"SMILES": smiles_list, "HADES_Score": scores})

    def score_csv(self, input_csv, output_csv, smiles_col="SMILES"):
        df_in = pd.read_csv(input_csv)
        df_out = df_in.merge(self.score(df_in[smiles_col].tolist()), left_on=smiles_col, right_on="SMILES", how="left")
        df_out.to_csv(output_csv, index=False)
        print(f"Wrote {len(df_out)} scored rows to {output_csv}")
        return df_out


def main():
    p = argparse.ArgumentParser(description="Score SMILES with HADES. No args = demo.")
    p.add_argument("-i", "--input", dest="input_csv", help="Path to input CSV")
    p.add_argument("-o", "--output", dest="output_csv", help="Path to write output CSV")
    p.add_argument("--smiles-col", default="SMILES")
    args = p.parse_args()
    hades = HADES_score()
    if args.input_csv and args.output_csv:
        hades.score_csv(args.input_csv, args.output_csv, smiles_col=args.smiles_col)
    elif args.input_csv or args.output_csv:
        p.error("both -i/--input and -o/--output are required together")
    else:
        print(hades.score(["NC(=O)N1CCN(C(=O)Cc2ccc(CN3CCc4ccccc43)cc2)CC1",
                            "Cc1cccnc1NC(=O)c1cccc(C(=O)N2CCN(C)CC2)c1", "C1CCCC1", "CCO"]))

if __name__ == "__main__":
    main()