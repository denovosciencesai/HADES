from HADES.quiet_noise import suppress_output
from HADES.featurize import Featurizer
from HADES.correlation_drop import CorrDrop
from HADES.hyperparameter_optimization import HPOptimizer
from HADES.param_spaces import default_param_spaces

import argparse, datetime as dt, json, os, time, importlib.util
from collections import Counter

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "XGBClassifier": XGBClassifier,
    "CatBoostClassifier": CatBoostClassifier,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}

def _step(label):
    """Announce a pipeline stage and return its start time."""
    print(f"  {label} ...", flush=True)
    return time.time()

def _done(t0, note):
    """Report how long the stage took, and what it produced."""
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"      -> {note}  [{mins}m {secs}s]" if mins else f"      -> {note}  [{secs}s]", flush=True)

def _load_param_spaces(path, use_gpu):
    """Load a custom param-spaces file. It must define a
    `default_param_spaces(use_gpu)` or `custom_param_spaces(use_gpu)` function
    (same signature as the built-in one), or a plain `PARAM_SPACES` dict."""
    spec = importlib.util.spec_from_file_location("custom_param_spaces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("default_param_spaces", "custom_param_spaces"):
        if hasattr(module, name):
            return getattr(module, name)(use_gpu)
    if hasattr(module, "PARAM_SPACES"):
        return module.PARAM_SPACES
    raise AttributeError(
        f"'{path}' must define `default_param_spaces(use_gpu)`, "
        f"`custom_param_spaces(use_gpu)`, or a `PARAM_SPACES` dict."
    )

class HADESTrainer:
    """
    End-to-end trainer for a HADES model ensemble: loads labeled SMILES data,
    featurizes it, drops correlated/uninformative descriptors, runs hyperparameter
    optimization for each model in MODEL_REGISTRY, fits the final ensemble, and
    persists everything (models, train/test splits, HPO results, and run config)
    needed to later score new molecules with `HADES_score`.
    """
    def __init__(self, output_dir=None, smiles_col="SMILES", label_col="phase",
                 positive_label=4, negative_label=0, test_size=0.2, random_state=42,
                 use_gpu=False, param_spaces=None, overwrite=False):

        """
        Args:
            output_dir (str, optional): Directory to write run outputs to. Defaults
                to a timestamped `hades_run_<YYYYmmdd_HHMMSS>` directory.
            smiles_col (str): Name of the column containing SMILES strings.
            label_col (str): Name of the column containing class labels used to
                derive the binary training target.
            positive_label: Value in `label_col` treated as the positive class.
            negative_label: Value in `label_col` treated as the negative class.
                If None, every row not equal to `positive_label` is treated as
                negative (no filtering is applied).
            test_size (float): Fraction of data held out for the test split.
            random_state (int): Seed used for splitting, feature selection, and
                model training.
            use_gpu (bool): Passed to the default param spaces to enable GPU
                training (e.g. XGBoost on CUDA), if `param_spaces` is not given.
            param_spaces (dict, optional): Custom hyperparameter search spaces,
                keyed by model class. Defaults to `default_param_spaces(use_gpu)`.
            overwrite (bool): If True, allows writing into an existing
                `output_dir`. Defaults to False.
        """
        self.output_dir = output_dir or f"hades_run_{dt.datetime.now():%Y%m%d_%H%M%S}"
        self.smiles_col, self.label_col = smiles_col, label_col
        self.positive_label, self.negative_label = positive_label, negative_label
        self.test_size, self.random_state = test_size, random_state
        self.overwrite = overwrite
        self.param_spaces = param_spaces or default_param_spaces(use_gpu)
        with suppress_output():
            self.featurizer = Featurizer()

    def _make_dirs(self):
        """
        Creates the output directory (and its `models` subdirectory), removing any
        model files an earlier run left in `models`.
        Raises:
            FileExistsError: If `self.output_dir` already exists and
                `self.overwrite` is False.
        Returns:
            str: The output directory path.
        """
        self._check_output_dir()
        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        # HADES_score loads every .pkl in models/, so models left from an earlier run
        # in this folder would be averaged in with the new ones. Their SHA256SUMS
        # would no longer describe the directory either.
        for name in os.listdir(models_dir):
            if name.endswith(".pkl") or name == "SHA256SUMS":
                os.remove(os.path.join(models_dir, name))
        return self.output_dir

    def _check_output_dir(self):
        """
        Raises:
            FileExistsError: If `self.output_dir` already exists and
                `self.overwrite` is False.
        """
        if os.path.exists(self.output_dir) and not self.overwrite:
            raise FileExistsError(f"'{self.output_dir}' exists; pass overwrite=True "
                                  f"(--overwrite on the command line).")

    def _load_data(self, csv_path):
        """
        Loads and validates the training CSV, deriving a binary `_target` column.
        Args:
            csv_path (str): Path to the input training CSV.
        Raises:
            KeyError: If `smiles_col` or `label_col` is missing from the CSV.
            ValueError: If fewer than two classes remain in `_target` after
                filtering.
        Returns:
            pd.DataFrame: The (possibly filtered) DataFrame with the added
                `_target` column.
        """
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
        """
        Computes molecular descriptors for every SMILES in `df`.
        Featurizes `df[self.smiles_col]` in bulk. No columns are dropped here:
        which descriptors survive is decided after the train/test split, from
        the training rows alone (see `_surviving_cols`).
        Args:
            df (pd.DataFrame): DataFrame containing the SMILES column to
                featurize.
        Returns:
            pd.DataFrame: Raw feature matrix with string-typed column names,
                indexed to match the rows that were successfully featurized.
        """
        with suppress_output():
            feats = self.featurizer.featurize_many_smiles(df[self.smiles_col])
        feats.columns = feats.columns.astype(str)
        return feats

    def _surviving_cols(self, X_train):
        """
        Returns the descriptors that survive the uninformative-column filters,
        computed from the training rows only.
        Both filters are row-dependent -- `drop_na_cols` keys on column dtype, so
        a single Mordred error object makes a whole column object-typed, and
        `drop_same_val_cols` keys on how often the modal value repeats. Running
        them over the full dataset would let the held-out rows help decide the
        feature set, so they are fitted on the training split alone.
        Args:
            X_train (pd.DataFrame): Training feature matrix, as returned by
                `_featurize`.
        Returns:
            list[str]: The column names to keep.
        """
        kept = self.featurizer.drop_same_val_cols(self.featurizer.drop_na_cols(X_train))
        return list(kept.columns)

    @staticmethod
    def _select_features(X, y, random_state):
        """
        Selects a feature subset by dropping highly-correlated descriptors.
        Args:
            X (pd.DataFrame): Training feature matrix.
            y: Training labels.
            random_state (int): Seed for the underlying RandomForestClassifier.
        Returns:
            tuple[list, float]: The list of column names to keep, and the
                optimal correlation threshold used to select them.
        """
        corrdrop = CorrDrop(RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=random_state))
        thresh, thresh_std = corrdrop.get_corr_threshold(X, y)
        keep_cols, opt_threshold = corrdrop.get_cols_to_keep(X, thresh, thresh_std)
        return list(keep_cols), float(opt_threshold)

    def _train_ensemble(self, X, y, results):
        """
        Fits one model per entry in `results` using its best hyperparameters.
        Args:
            X: Training feature matrix.
            y: Training labels (coerced to int).
            results (dict): Mapping of model name -> {"best_params": dict, ...},
                as produced by `HPOptimizer.param_search`.
        Returns:
            dict[str, object]: Mapping of model name to its fitted estimator.
        """
        y = list(map(int, y))
        weights = compute_sample_weight("balanced", y=y)
        counts = Counter(y)
        cb_weights = [len(y) / counts[c] for c in sorted(counts)]
        # HPO ran each model with its param space's extra_args, but best_params does
        # not carry them. Without merging them back the final fit is not the model
        # that was selected -- n_jobs, verbose and device settings would all be lost.
        extra_args = {cls.__name__: spec.get("extra_args", {}) for cls, spec in self.param_spaces.items()}
        models = {}
        for name, ctor in MODEL_REGISTRY.items():
            if name not in results:
                # Allows a custom param_spaces file to train only a subset
                # of MODEL_REGISTRY's models.
                continue
            best = {**extra_args.get(name, {}), **results[name]["best_params"]}
            best.setdefault("random_seed" if name == "CatBoostClassifier" else "random_state", self.random_state)
            if name == "CatBoostClassifier":
                best.setdefault("verbose", 0)
                models[name] = ctor(class_weights=cb_weights, **best).fit(X, y)
            elif name == "XGBClassifier":
                best.setdefault("eval_metric", "logloss")
                models[name] = ctor(**best).fit(X, y, sample_weight=weights)
            else:
                models[name] = ctor(**best).fit(X, y, sample_weight=weights)
        return models

    def run(self, csv_path, reuse_hpo=False):
        """
        Runs the full training pipeline end to end and writes all outputs to disk.
        Args:
            csv_path (str): Path to the input training CSV.
            reuse_hpo (bool): If True and a `hyperparameters.json` already exists
                in the output directory, loads those results instead of re-running
                hyperparameter search. Defaults to False.
        Returns:
            str: The output directory containing all run artifacts.
        """
        # Validate before creating anything, so a bad CSV or column name does not
        # leave behind a folder that blocks the retry.
        self._check_output_dir()
        t0 = _step("1/6 Loading data")
        df = self._load_data(csv_path)
        out = self._make_dirs()
        n_pos = int(df["_target"].sum())
        _done(t0, f"{len(df)} molecules ({n_pos} positive, {len(df) - n_pos} negative)")

        t0 = _step("2/6 Featurizing (Mordred + ADMET-AI + QED)")
        feats = self._featurize(df)
        dropped = len(df) - len(feats)
        note = f"{len(feats)} molecules, {feats.shape[1]} raw descriptors"
        _done(t0, note + (f" -- {dropped} SMILES could not be parsed" if dropped else ""))
        y, smiles = df["_target"].loc[feats.index], df[self.smiles_col].loc[feats.index]

        X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
            feats, y, smiles, test_size=self.test_size, random_state=self.random_state, stratify=y)
        print(f"  3/6 Split ......... {len(X_train)} train / {len(X_test)} test", flush=True)

        # Fitted on X_train
        surviving = self._surviving_cols(X_train)
        X_train, X_test = X_train[surviving], X_test[surviving]

        t0 = _step(f"4/6 Selecting features from {X_train.shape[1]} descriptors (slowest step)")
        keep_cols, opt_threshold = self._select_features(X_train, y_train, self.random_state)
        X_train = X_train[keep_cols].astype(float)
        # A descriptor can be numeric across every training row and still hold a
        # Mordred error object on a test molecule, which astype(float) raises on.
        # Coerce those to NaN and fill from the training means -- the same
        # transform HADES_CALCULATOR applies to new molecules when scoring.
        X_test = X_test[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(X_train.mean())
        _done(t0, f"kept {len(keep_cols)} descriptors (correlation threshold {opt_threshold:.2f})")

        hpo_path = os.path.join(out, "hyperparameters.json")
        if reuse_hpo and os.path.exists(hpo_path):
            with open(hpo_path) as f:
                results = json.load(f)
            print(f"  5/6 Hyperparameter search ... reused {hpo_path}", flush=True)
        else:
            t0 = _step(f"5/6 Hyperparameter search over {len(self.param_spaces)} models")
            results = HPOptimizer(self.param_spaces, X_train, np.asarray(y_train)).param_search()
            with open(hpo_path, "w") as f:
                json.dump(results, f, indent=4)
            _done(t0, f"{len(results)} models tuned")

        t0 = _step("6/6 Fitting final ensemble")
        models = self._train_ensemble(X_train, y_train, results)
        for name, model in models.items():
            joblib.dump(model, os.path.join(out, "models", f"{name}.pkl"))
        _done(t0, f"{len(models)} models saved to {os.path.join(out, 'models')}")

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
    """
    Attempts to parse a label value as an int, falling back to the original
    string if conversion fails.
    Args:
        v (str): Raw label value, typically from the command line.
    Returns:
        int or str: The parsed integer, or the original string if it isn't
            numeric.
    """
    try:
        return int(v)
    except ValueError:
        return v

def main():
    """
    Command-line entry point for training a HADES ensemble.
    """
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
    p.add_argument("--param-spaces-file", default=None,
                    help="Path to a .py file defining default_param_spaces(use_gpu) or "
                         "custom_param_spaces(use_gpu) or a PARAM_SPACES dict, to override "
                         "the built-in HPO search spaces.")
    args = p.parse_args()

    negative = None if args.negative_label.lower() == "none" else _parse_label(args.negative_label)
    param_spaces = (_load_param_spaces(args.param_spaces_file, args.use_gpu)
                     if args.param_spaces_file else None)

    HADESTrainer(output_dir=args.output_dir, smiles_col=args.smiles_col, label_col=args.label_col,
                 positive_label=_parse_label(args.positive_label), negative_label=negative,
                 test_size=args.test_size, use_gpu=args.use_gpu, param_spaces=param_spaces,
                 overwrite=args.overwrite
                 ).run(args.input_csv, reuse_hpo=args.reuse_hpo)

if __name__ == "__main__":
    main()