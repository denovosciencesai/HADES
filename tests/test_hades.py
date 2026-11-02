"""A few checks that HADES still gives the right answers.

Deliberately small. The point is to catch two things that have gone wrong
before and would otherwise go unnoticed: a silent change in the scores, and
rows losing track of the molecule they came from.

Runs from any working directory: HADES resolves its models and descriptor
spec relative to the installed package, not to the caller's cwd.

    pytest
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Before pandas: importing pandas first loads an older libstdc++ than RDKit needs.
import HADES_CALCULATOR
from HADES_CALCULATOR import HADES_score

import pandas as pd

# Repository-only: the paper's training split is not shipped with the package,
# so these tests reach for it by path instead of assuming a working directory.
X_TRAIN = str(Path(__file__).resolve().parent.parent / "train_test_saved" / "X_train.parquet")

# Measured with the models in HADES/data/models/, whose exact bytes SHA256SUMS
# records. If one of these moves, the featurizer, the models or the descriptor
# set changed -- update these numbers only when that was the intention.
GOLDEN = {
    "CCO": 0.017846526109747365,
    "CCCC": 0.00045819962966435495,
    "c1ccccc1": 0.011470215497587182,
}

# Loose enough to survive a different BLAS or machine, far tighter than any
# real change to the model would be.
TOLERANCE = 1e-6


@pytest.fixture(scope="session")
def hades():
    """One ensemble for the whole session; loading it costs ~2s.

    `verify="require"` means every test also fails if the model files do not
    match SHA256SUMS.
    """
    return HADES_score(verify="require")


@pytest.mark.parametrize("smiles, expected", sorted(GOLDEN.items()))
def test_score_has_not_drifted(hades, smiles, expected):
    assert hades.score(smiles) == pytest.approx(expected, abs=TOLERANCE)


def test_featurizer_preserves_caller_index(hades):
    """Rows must stay matched to the molecules they came from.

    Regression test for 59a3f3c, where an unparseable SMILES shifted every
    later row and scores came back attached to the wrong molecule.
    """
    feats = hades.featurizer.featurize_many_smiles(["CCO", "NOT_A_SMILES", "CCCC"])
    assert list(feats.index) == [0, 2]


def test_scoring_a_list_skips_what_it_cannot_parse(hades):
    df = hades.score(["CCO", "NOT_A_SMILES", "CCCC"])
    assert df["SMILES"].tolist() == ["CCO", "CCCC"]
    assert df["HADES_Score"].tolist() == pytest.approx(
        [GOLDEN["CCO"], GOLDEN["CCCC"]], abs=TOLERANCE)


def test_scoring_one_bad_smiles_raises(hades):
    with pytest.raises(ValueError):
        hades.score("NOT_A_SMILES")


def test_piped_output_is_not_duplicated():
    """Text printed before scoring must reach a pipe exactly once.

    Scoring forks worker processes. With stdout piped (so block-buffered),
    each child inherited the parent's unflushed output and wrote it again on
    exit, so anything piped into tee or a log file came out doubled.
    Runs in a fresh interpreter because pytest captures stdout itself.
    """
    script = (
        "from HADES_CALCULATOR import HADES_score\n"
        "h = HADES_score()\n"
        "print('MARKER-ONCE')\n"
        "h.score(['CCO', 'CCCC'])\n"
    )
    out = subprocess.run([sys.executable, "-c", script], stdout=subprocess.PIPE,
                         text=True, check=True).stdout
    assert out.count("MARKER-ONCE") == 1


def _write_x_train(tmp_path, reorder):
    df = pd.read_parquet(X_TRAIN)
    path = tmp_path / "X_train.parquet"
    df[reorder(list(df.columns))].to_parquet(path)
    return str(path)


def test_smiles_column_is_found_by_name(hades, tmp_path):
    """SMILES need not be the last column: it is dropped by name, not position."""
    path = _write_x_train(tmp_path, lambda cols: cols[-1:] + cols[:-1])
    assert HADES_score(x_train_parquet=path).descriptor_cols == hades.descriptor_cols


def test_missing_smiles_column_raises(tmp_path):
    path = _write_x_train(tmp_path, lambda cols: [c for c in cols if c != "SMILES"])
    with pytest.raises(ValueError, match="no 'SMILES' column"):
        HADES_score(x_train_parquet=path)


def test_descriptors_not_matching_the_models_raise(tmp_path):
    """A different descriptor order would misalign every feature fed to the models."""
    path = _write_x_train(tmp_path, lambda cols: cols[1::-1] + cols[2:])
    with pytest.raises(ValueError, match="do not match"):
        HADES_score(x_train_parquet=path)


def test_score_csv_keeps_every_row_once_in_order(hades, tmp_path):
    """Repeated SMILES must not multiply rows; bad and blank cells stay, unscored."""
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("SMILES,name\nCCO,a\nCCO,b\nNOT_A_SMILES,c\n,d\nCCCC,e\n")
    out = hades.score_csv(str(src), str(dst))
    assert out["name"].tolist() == ["a", "b", "c", "d", "e"]
    assert pd.read_csv(dst)["name"].tolist() == ["a", "b", "c", "d", "e"]
    assert out["HADES_Score"].tolist()[:2] == pytest.approx([GOLDEN["CCO"]] * 2, abs=TOLERANCE)
    assert out["HADES_Score"].iloc[2:4].isna().all()
    assert out["HADES_Score"].iloc[4] == pytest.approx(GOLDEN["CCCC"], abs=TOLERANCE)


def test_non_string_smiles_are_skipped(hades):
    df = hades.score(["CCO", float("nan"), None, "CCCC"])
    assert df["SMILES"].tolist() == ["CCO", "CCCC"]


def test_models_dir_without_models_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .pkl model files"):
        HADES_score(models_dir=str(tmp_path))


def test_packaged_spec_matches_the_training_split():
    """The shipped spec must still say what X_train.parquet says.

    It is derived from that file, so this is the check that keeps the two from
    drifting -- and the record of how the shipped file was produced.
    """
    packaged_cols, packaged_means = HADES_CALCULATOR._packaged_spec()
    train_cols, train_means = HADES_CALCULATOR._spec_from_x_train(X_TRAIN)
    assert packaged_cols == train_cols
    assert packaged_means.to_numpy() == pytest.approx(train_means.to_numpy())


def test_defaults_do_not_depend_on_the_working_directory(tmp_path):
    """HADES must score from any folder, not only from the repository root.

    Regression test for the packaging fix: the defaults were `./HADES_saved_models`
    and `./train_test_saved/X_train.parquet`, so an installed copy found neither
    unless the caller happened to be standing in the repository.
    """
    script = ("from HADES_CALCULATOR import HADES_score\n"
              "print(HADES_score(verify='require').score('CCO'))\n")
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                         stdout=subprocess.PIPE, text=True, check=True).stdout
    assert float(out.strip()) == pytest.approx(GOLDEN["CCO"], abs=TOLERANCE)
