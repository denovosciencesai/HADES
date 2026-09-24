"""End-to-end check that training produces a model HADES_score can load.

Slow -- about 16 minutes even on the 20 bundled molecules, because feature
selection has a large fixed cost (see "How long training takes" in the README).
It is excluded by default; run it with:

    pytest -m slow
"""

import pytest

# Before pandas: importing pandas first loads an older libstdc++ than RDKit needs.
from HADES_CALCULATOR import HADES_score
from train_your_HADES import HADESTrainer
from parameters_test import custom_param_spaces


@pytest.fixture(scope="module")
def trainer_factory():
    """Building a trainer loads the featurizer (~seconds), so share one."""
    base = HADESTrainer()
    def make(output_dir, overwrite=False, label_col="phase"):
        t = HADESTrainer.__new__(HADESTrainer)
        t.__dict__.update(base.__dict__, output_dir=str(output_dir), overwrite=overwrite,
                          label_col=label_col)
        return t
    return make


def test_bad_input_leaves_no_output_folder(trainer_factory, tmp_path):
    """A typo in the column name must not leave a folder that blocks the retry."""
    out = tmp_path / "run"
    with pytest.raises(KeyError):
        trainer_factory(out, label_col="nope").run("data_to_test.csv")
    assert not out.exists()


def test_existing_folder_needs_overwrite(trainer_factory, tmp_path):
    with pytest.raises(FileExistsError, match="--overwrite"):
        trainer_factory(tmp_path).run("data_to_test.csv")


def test_overwrite_removes_models_from_an_earlier_run(trainer_factory, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("CatBoostClassifier.pkl", "SHA256SUMS", "notes.txt"):
        (models / name).write_text("stale")
    trainer_factory(tmp_path, overwrite=True)._make_dirs()
    assert sorted(p.name for p in models.iterdir()) == ["notes.txt"]


@pytest.mark.slow
def test_trained_model_can_be_loaded_and_scores(tmp_path):
    out = HADESTrainer(output_dir=str(tmp_path / "run"), smiles_col="SMILES", label_col="phase",
                       positive_label=4, negative_label=0,
                       param_spaces=custom_param_spaces()).run("data_to_test.csv")

    # A fresh run has no SHA256SUMS, so there is nothing to verify against.
    hades = HADES_score(models_dir=f"{out}/models", x_train_parquet=f"{out}/X_train.parquet",
                        verify="off")
    assert len(hades.models) == 5
    df = hades.score(["CCCC", "CCO", "c1ccccc1"])
    assert df["SMILES"].tolist() == ["CCCC", "CCO", "c1ccccc1"]
    assert df["HADES_Score"].between(0, 1).all()
