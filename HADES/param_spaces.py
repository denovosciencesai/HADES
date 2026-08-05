from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier

def default_param_spaces(use_gpu: bool = False) -> dict:
    xgb_extra = {"seed": 42}
    if use_gpu:
        xgb_extra["device"] = "cuda"
    return {
        RandomForestClassifier: {
            "params": {
                "n_estimators": ("int", 200, 1500, 100),
                "max_features": ("categorical", ["sqrt", "log2", 100, 200, 300]),
                "max_depth": ("int", 2, 50, 2),
                "criterion": ("categorical", ["gini", "entropy", "log_loss"]),
                "min_samples_split": ("int", 2, 20, 2),
                "bootstrap": ("categorical", [True, False]),
            },
            "extra_args": {"n_jobs": -1, "random_state": 42},
        },
        HistGradientBoostingClassifier: {
            "params": {
                "max_iter": ("int", 200, 1500, 100),
                "max_features": ("float", 0.3, 1.0),
                "learning_rate": ("loguniform", 0.01, 0.5),
                "max_depth": ("int", 2, 10, 1),
                "max_bins": ("int", 63, 255, 16),
                "min_samples_leaf": ("int", 1, 101, 2),
            },
            "extra_args": {"random_state": 42},
        },
        XGBClassifier: {
            "params": {
                "n_estimators": ("int", 200, 1500, 100),
                "learning_rate": ("loguniform", 0.01, 0.5),
                "max_depth": ("int", 2, 10, 1),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.3, 1.0),
            },
            "extra_args": xgb_extra,
        },
        CatBoostClassifier: {
            "params": {
                "iterations": ("int", 100, 5000, 100),
                "learning_rate": ("loguniform", 0.01, 0.5),
                "depth": ("int", 2, 10, 1),
                "l2_leaf_reg": ("loguniform", 1, 5),
                "border_count": ("int", 32, 128, 32),
            },
            "extra_args": {"verbose": 0, "random_seed": 42},
        },
        LGBMClassifier: {
            "params": {
                "n_estimators": ("int", 200, 1500, 100),
                "learning_rate": ("loguniform", 0.01, 0.5),
                "max_depth": ("int", 2, 10, 1),
                "num_leaves": ("int", 30, 100, 10),
                "min_data_in_leaf": ("int", 5, 50, 5),
            },
            "extra_args": {"verbose": -1, "n_jobs": -1, "random_state": 42},
        },
    }