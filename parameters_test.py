# small_param_spaces.py
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def default_param_spaces(use_gpu: bool = False) -> dict:
    return {
        RandomForestClassifier: {
            "params": {
                "n_estimators": ("int", 50, 150, 50),
                "max_depth": ("int", 2, 10, 2),
            },
            "extra_args": {"n_jobs": -1, "random_state": 42},
        },
        HistGradientBoostingClassifier: {
            "params": {
                "max_iter": ("int", 50, 150, 50),
                "learning_rate": ("loguniform", 0.05, 0.3),
                "max_depth": ("int", 2, 6, 1),
            },
            "extra_args": {"random_state": 42},
        },
        XGBClassifier: {
            "params": {
                "n_estimators": ("int", 50, 150, 50),
                "learning_rate": ("loguniform", 0.05, 0.3),
                "max_depth": ("int", 2, 6, 1),
            },
            "extra_args": {"seed": 42},
        },
        CatBoostClassifier: {
            "params": {
                "iterations": ("int", 50, 150, 50),
                "learning_rate": ("loguniform", 0.05, 0.3),
                "depth": ("int", 2, 6, 1),
            },
            "extra_args": {"verbose": 0, "random_seed": 42},
        },
        LGBMClassifier: {
            "params": {
                "n_estimators": ("int", 50, 150, 50),
                "learning_rate": ("loguniform", 0.05, 0.3),
                "max_depth": ("int", 2, 6, 1),
            },
            "extra_args": {"verbose": -1, "n_jobs": -1, "random_state": 42},
        },
    }