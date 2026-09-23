"""A few checks that HADES still gives the right answers.

Deliberately small. The point is to catch two things that have gone wrong
before and would otherwise go unnoticed: a silent change in the scores, and
rows losing track of the molecule they came from.

Run from the repository root, so the default `./HADES_saved_models` and
`./train_test_saved` paths resolve:

    pytest
"""

import pytest

from HADES_CALCULATOR import HADES_score

# Measured with the models in HADES_saved_models/, whose exact bytes SHA256SUMS
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
