import argparse, datetime as dt, json, os
from collections import Counter

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from HADES.quiet_noise import suppress_output
from HADES.featurize import Featurizer
from HADES.correlation_drop import CorrDrop
from HADES.ml_modeling import HPOptimizer
from HADES.param_spaces import default_param_spaces

MODEL_REGISTRY = {
    "XGBClassifier": XGBClassifier,
    "CatBoostClassifier": CatBoostClassifier,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}


class HADESTrainer:
    def __init__(self, output_dir=None, smiles_col="SMILES", label_col="phase",
                 positive_label=4, negative_label=0, test_size=0.2, random_state=42,
                 use_gpu=False, param_spaces=None, overwrite=False):
        self.output_dir = output_dir or f"hades_run_{dt.datetime.now():%Y%m%d_%H%M%S}"
        self.smiles_col, self.label_col = smiles_col, label_col
        self.positive_label, self.negative_label = positive_label, negative_label
        self.test_size, self.random_state = test_size, random_state
        self.overwrite = overwrite
        self.param_spaces = param_spaces or default_param_spaces(use_gpu)
        with suppress_output():
            self.featurizer = Featurizer()

    def _make_dirs(self):
        if os.path.exists(self.output_dir) and not self.overwrite:
            raise FileExistsError(f"'{self.output_dir}' exists; pass overwrite=True.")
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        return self.output_dir

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        for c in (self.smiles_col, self.label_col):
            if c not in df.columns:
                raise KeyError(f"Column '{c}' not in {csv_path}. Available: {list(df.columns)}")
        if self.negative_label is not None:
            df = df[df[self.label_col].isin([self.positive_label, self.negative_label])].copy()
        df["_target"] = (df[self.label_col] == self.positive_label).astype(int)
        if df["_target"].nunique() < 2:
            raise ValueError("Need both classes present after filtering.")
        return df

    def _featurize(self, df):
        with suppress_output():
            feats = self.featurizer.featurize_many_smiles(df[self.smiles_col])
        feats = self.featurizer.drop_same_val_cols(self.featurizer.drop_na_cols(feats))
        feats.columns = feats.columns.astype(str)
        return feats

    @staticmethod
    def _select_features(X, y, random_state):
        corrdrop = CorrDrop(RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=random_state))
        thresh, thresh_std = corrdrop.get_corr_threshold(X, y)
        keep_cols, opt_threshold = corrdrop.get_cols_to_keep(X, thresh, thresh_std)
        return list(keep_cols), float(opt_threshold)

    def _train_ensemble(self, X, y, results):
        y = list(map(int, y))
        weights = compute_sample_weight("balanced", y=y)
        counts = Counter(y)
        cb_weights = [len(y) / counts[c] for c in sorted(counts)]
        models = {}
        for name, ctor in MODEL_REGISTRY.items():
            best = dict(results[name]["best_params"])
            best.setdefault("random_seed" if name == "CatBoostClassifier" else "random_state", self.random_state)
            if name == "CatBoostClassifier":
                models[name] = ctor(class_weights=cb_weights, verbose=0, **best).fit(X, y)
            elif name == "XGBClassifier":
                models[name] = ctor(eval_metric="logloss", **best).fit(X, y, sample_weight=weights)
            else:
                models[name] = ctor(**best).fit(X, y, sample_weight=weights)
        return models

    def run(self, csv_path, reuse_hpo=False):
        out = self._make_dirs()
        df = self._load_data(csv_path)
        feats = self._featurize(df)
        y, smiles = df["_target"].loc[feats.index], df[self.smiles_col].loc[feats.index]

        X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
            feats, y, smiles, test_size=self.test_size, random_state=self.random_state, stratify=y)

        keep_cols, opt_threshold = self._select_features(X_train, y_train, self.random_state)
        X_train, X_test = X_train[keep_cols].astype(float), X_test[keep_cols].astype(float)

        hpo_path = os.path.join(out, "hyperparameters.json")
        if reuse_hpo and os.path.exists(hpo_path):
            with open(hpo_path) as f:
                results = json.load(f)
        else:
            results = HPOptimizer(self.param_spaces, X_train, np.asarray(y_train)).param_search()
            with open(hpo_path, "w") as f:
                json.dump(results, f, indent=4)

        models = self._train_ensemble(X_train, y_train, results)
        for name, model in models.items():
            joblib.dump(model, os.path.join(out, "models", f"{name}.pkl"))

        xtr, xte = X_train.copy(), X_test.copy()
        xtr["SMILES"], xte["SMILES"] = smi_train.values, smi_test.values
        xtr.to_parquet(os.path.join(out, "X_train.parquet"))
        xte.to_parquet(os.path.join(out, "X_test.parquet"))
        y_train.to_csv(os.path.join(out, "y_train.csv"))
        y_test.to_csv(os.path.join(out, "y_test.csv"))

        config = {"created": dt.datetime.now().isoformat(timespec="seconds"), "smiles_col": self.smiles_col,
                  "label_col": self.label_col, "positive_label": self.positive_label,
                  "negative_label": self.negative_label, "opt_threshold": opt_threshold,
                  "descriptor_cols": keep_cols, "model_names": list(models.keys())}
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print(f"Done. Score with: HADES_score(models_dir='{out}/models', x_train_parquet='{out}/X_train.parquet')")
        return out


def _parse_label(v):
    try:
        return int(v)
    except ValueError:
        return v


def main():
    p = argparse.ArgumentParser(description="Train a HADES ensemble.")
    p.add_argument("-i", "--input", dest="input_csv", required=True)
    p.add_argument("-o", "--output-dir", default=None)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="phase")
    p.add_argument("--positive-label", default="4")
    p.add_argument("--negative-label", default="0", help="Use 'none' for everything-else-is-negative.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--reuse-hpo", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    negative = None if args.negative_label.lower() == "none" else _parse_label(args.negative_label)
    HADESTrainer(output_dir=args.output_dir, smiles_col=args.smiles_col, label_col=args.label_col,
                 positive_label=_parse_label(args.positive_label), negative_label=negative,
                 test_size=args.test_size, use_gpu=args.use_gpu, overwrite=args.overwrite
                 ).run(args.input_csv, reuse_hpo=args.reuse_hpo)


if __name__ == "__main__":
    main()