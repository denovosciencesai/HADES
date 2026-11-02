![HADES](hades_logo.svg)

# HADES — Holistic AI-based Drug-likeness Estimation Score

HADES (Holistic AI-based Drug-likeness Estimation Score) is a soft-voting ensemble of five ML models (HistGradientBoosting, RandomForest, CatBoost, XGBoost, and LightGBM) that predicts the oral drug-likeness of a molecule from its SMILES string.

You can score molecules with the provided pretrained HADES model (https://doi.org/10.1021/acs.jcim.5c02953), or train your own model on your own labelled compounds.

## Requirements

- Python 3.10
- Conda (Miniconda or Anaconda) recommended — the project depends on RDKit and
  other compiled libraries that install most reliably via conda-forge.

## Setup

```bash
conda env create -f environment.yml
conda activate hades
```

That installs Python 3.10, RDKit (via conda-forge), and every pinned package in
`requirements.txt`.

Optionally, install the project into the environment as well:

```bash
pip install -e .
```

That makes `from HADES import HADES_score, HADESTrainer` importable from any
directory, and adds two commands, `hades-score` and `hades-train`, which take the
same options as `python HADES_CALCULATOR.py` and `python train_your_HADES.py`. The
trained models and the descriptor spec are installed alongside the code, so both
work from any directory -- pass `--models-dir` only to score with an ensemble of
your own.

## Repository layout

```
.
├── HADES ┐                                     # core package
          ├──__init__.py                        # package marker; re-exports HADES_score and HADESTrainer
          ├──correlation_drop.py                # drops highly correlated features
          ├──featurize.py                       # featurizes given SMILES with Mordred, ADMET and QED features
          ├──hyperparameter_optimization.py     # finds best hyperparameters using the Optuna algorithm (https://github.com/optuna/optuna)
          ├──integrity.py                       # writes and checks the SHA-256 manifest of a model directory
          ├──param_spaces.py                    # hyperparameter search space for Optuna
          ├──quiet_noise.py                     # silences warnings, tqdm bars, etc. for a cleaner console; can be disabled for debugging
          └──data/                              # shipped with the package, so an installed HADES finds it anywhere
                ├──models/                      # the trained ML models (.pkl files) for HADES_score
                │     └──SHA256SUMS             # SHA-256 digest of each model file, checked before loading
                └──feature_spec.parquet         # descriptor order and means, derived from train_test_saved/X_train.parquet
├── train_test_saved/                           # split data used in the original paper (not shipped with the package)
├── HADES_CALCULATOR.py                         # script to calculate HADES_score from the terminal or a Jupyter notebook, for single molecules or batches
├── train_your_HADES.py                         # script to train your own HADES model on your own data
├── parameters_test.py                          # a small hyperparameter set for testing custom parameter spaces during HADES training
├── tests/                                      # pytest suite: golden scores and regression checks
├── environment.yml                             # defines the HADES conda environment
├── hades_logo.svg                              # Logo for Repository
├── pyproject.toml                              # packaging metadata (`pip install -e .`)
├── requirements.txt                            # required libraries, packages, etc.
├── CITATION.cff                                # how to cite HADES and the paper
├── CONTRIBUTING.md                             # how to set up, test and submit changes
├── SECURITY.md                                 # how to report a vulnerability privately
├── data_to_test.csv                            # small sample dataframe for testing the project
└── README.md                                   # description of the project and its usage
```

---

## Model artifacts and integrity

The five pretrained models ship as Python pickles (`HADES/data/models/*.pkl`).
**Unpickling a file executes code contained in it.** Only load model files from a
source you trust — this repository, or a training run you produced yourself. That
applies in particular to section 3 below, where you point `--models-dir` at a
directory of models: if the directory came from somewhere else, you are running
whatever is in it.

### Published checksums

`HADES/data/models/SHA256SUMS` records the SHA-256 digest of every shipped model:

```
e990b6cdd5c83fbc6c186b23df325c6a10d0b6969261f5dc9a7376f493a6e3ec  CatBoostClassifier.pkl
b8946b76986e65372135d152e69f088338cfba66a792c441b1b2da246e2afd19  HistGradientBoostingClassifier.pkl
c96a8d48d24e2ac6e58531ff6d8dc38d85058acc798ce90dceddb24bd0866dbf  LGBMClassifier.pkl
2d5fbebafa31241202d59cc3e1f5e82f79b1d07453cfc0c2e421d8f31fa26e83  RandomForestClassifier.pkl
10314d71ccf26a22df20d646bb8837d6cd163dab22d0cafd7c8d95c418bd877d  XGBClassifier.pkl
```

It is a standard `sha256sum` file, so it can be checked without Python:

```bash
cd HADES/data/models && sha256sum -c SHA256SUMS
```

### Checking on load

`HADES_score` checks the models against that manifest before unpickling anything:

| `verify=` | Behaviour |
| --- | --- |
| `"auto"` (default) | Verify against `SHA256SUMS` when the model directory has one; warn on stderr when it does not. |
| `"require"` | A missing manifest — or a `.pkl` the manifest does not list — is an error. |
| `"off"` | Skip the check. |

A mismatch raises `ChecksumError` and nothing is loaded. The same choice exists on
the command line:

```bash
python HADES_CALCULATOR.py --verify require -i data_to_test.csv -o output.csv
```

### Your own trained models

Training does not write a manifest, so scoring with a model you trained yourself
prints a warning that the directory is unverified. To record its digests once:

```python
from HADES.integrity import write_manifest

write_manifest("quick_test_run/models")
```

After that the directory verifies like the shipped one.

### What this does and does not buy you

Checksums detect tampering and corruption, and they pin which bytes produced a
given set of scores. They do **not** make an untrusted pickle safe: a hostile file
whose digest matches the manifest still executes on load. Verification is only
meaningful if the manifest itself reached you through a channel you trust.

---

## 1. Score molecules with the provided model

### In a Jupyter notebook (or any Python session)

```python
from HADES_CALCULATOR import HADES_score

hades = HADES_score()
hades.score(['CCCCSCC', 'c1ccccc1'])
```

`score()` returns:

- a **DataFrame** (columns `SMILES`, `HADES_Score`) when you pass a list of SMILES,
- a single **float** when you pass one SMILES string, e.g. `hades.score("CCO")`.

Progress bars and library log messages are silenced by default. For debugging,
`HADES_score(quiet=False)` brings back the library log messages. Progress bars
stay off either way: importing `HADES.quiet_noise` disables `tqdm` for the whole
process, and nothing re-enables it.

### Invalid SMILES

RDKit cannot parse every string. What happens depends on how you call `score()`:

- **A single SMILES** raises `ValueError`, because there is no sensible number to
  return for one molecule that does not exist:

  ```python
  hades.score("NOT_A_SMILES")
  # ValueError: Cannot score 'NOT_A_SMILES': RDKit cannot parse it
  ```

- **A list** skips the unparseable ones, scores the rest, and reports what it
  skipped. The returned DataFrame contains only the molecules that were scored,
  so it can be shorter than the input:

  ```python
  hades.score(["CCCC", "NOT_A_SMILES", "CCO"])
  # Could not calculate 1 SMILES: ['NOT_A_SMILES']
  #    SMILES  HADES_Score
  #      CCCC     0.000458
  #       CCO     0.017847
  ```

  If nothing parses, you get an empty DataFrame rather than an error.

The report is printed even with the default `quiet=True`, so molecules never
disappear silently.

### From the command line (CSV in, CSV out)

To score a whole file and get a new CSV with the scores added:

```bash
python HADES_CALCULATOR.py -i data_to_test.csv -o output.csv --smiles-col SMILES
```

The output CSV is your input file plus a `HADES_Score` column. Use
`--smiles-col` to name the column that holds the SMILES if it is not `SMILES`.

Every input row is kept, in its original order, including repeated SMILES.
Rows whose SMILES could not be parsed, or whose SMILES cell is empty, get an
empty `HADES_Score`, and the run prints which ones they were.

---

## 2. Train your own model

If you have your own dataset of positive- and negative-labelled molecules, you
can train a new HADES ensemble on it.

**Input:** a CSV with a SMILES column and a label column marking which
compounds are positive and which are negative.

### How long training takes

Training is slow, and it is slow even on a handful of molecules. Measured on an
Intel Core 5 120U (10 cores, 12 threads):

| Compounds | Featurizing (2/6) | Feature selection (4/6) | Hyperparameter search (5/6) | Total |
| --- | --- | --- | --- | --- |
| 20 | 1s | 14m 18s | 1m 48s | **16m 17s** |
| 200 | 8s | 15m 58s | 2m 33s | **18m 46s** |
| 2000 | 1m 33s | 20m 56s | 11m 58s | **34m 37s** |

Two stages dominate, and which one depends on how much data you have.

**Step 4/6, feature selection**, sweeps 99 correlation thresholds, and at each one
runs 5-fold cross-validation with a 1000-tree random forest — 495 forest fits in
total. Most of that cost is fixed rather than proportional to your data: it takes
about 14 minutes on 20 molecules and still only 21 minutes on 2000. Treat roughly
**15 minutes as a floor you pay regardless of dataset size**.

**Step 5/6, hyperparameter search**, is the stage that actually scales — under 2
minutes at 20 compounds, 12 minutes at 2000, and still climbing. Above a few
thousand compounds, expect it to overtake feature selection. 

Featurizing (step 2/6) grows linearly with compound count and stays modest.

Scoring is cheap next to training: about 30 ms per molecule (200 compounds in
5.6s, 2000 in 60.4s), so 10,000 molecules take roughly five minutes.

### Command line

```bash
python train_your_HADES.py -i data_to_test.csv -o output_dir \
    --smiles-col SMILES \
    --label-col phase \
    --positive-label 4 \
    --negative-label 0
```

`-o output_dir` names the output folder. Everything for the run is written inside that folder:

```
output_dir/
├── models                 # Saved ML models
├── config.json            # settings + the exact feature columns
├── hyperparameters.json   # best parameters found
├── X_test.parquet         # saved test set
├── X_train.parquet        # saved train set
├── y_test.csv             # saved test-set target values
└── y_train.csv            # saved train-set target values
```

Because each run has its own folder, training a new model never overwrites a
previous one: if the folder already exists, training stops with an error. Pass
`--overwrite` (`overwrite=True` in Python) to train into an existing folder on
purpose. That replaces the models in it -- any `.pkl` files an earlier run left in
`models/` are deleted first, so the two runs are never mixed.

To retrain without repeating the hyperparameter search, pass `--reuse-hpo`
together with `--overwrite`. Step 5/6 then loads `hyperparameters.json` from the
earlier run in that folder instead of searching again.

Molecules whose SMILES cannot be parsed are dropped before training, and step
2/6 reports how many. Labels stay matched to the molecules that survived.

### In a Jupyter notebook

```python
from train_your_HADES import HADESTrainer

HADESTrainer(
    output_dir="my_model",
    smiles_col="SMILES",
    label_col="phase",
    positive_label=4,
    negative_label=0,
).run("data_to_test.csv")
```

### Customize the hyperparameter search (optional)

If you want to run the hyperparameter search over a new parameter space, you can customize it as well.

To use this from bash, set up a new `.py` file structured like `HADES/param_spaces.py` and pass it in on the command line. `parameters_test.py` is provided as a ready-made example you can use when training your own model:

```bash
python train_your_HADES.py -i data_to_test.csv -o quick_test_run --param-spaces-file parameters_test.py --overwrite
```
```python
from train_your_HADES import HADESTrainer
from parameters_test import custom_param_spaces

small_param_spaces = custom_param_spaces()  # or write/import your own parameters
trainer = HADESTrainer(
    output_dir="quick_test_run",
    smiles_col="SMILES",
    label_col="phase",
    param_spaces=small_param_spaces,
    overwrite=True,
)
out_dir = trainer.run("data_to_test.csv")
```
---

## 3. Use a model you trained

After training your own model, you can calculate a custom HADES_score from a Jupyter notebook or the terminal. Point it at the output folder your run produced — `quick_test_run` below is the one created by the `--param-spaces-file` example above.

```bash
python HADES_CALCULATOR.py --models-dir quick_test_run/models --x-train quick_test_run/X_train.parquet -i data_to_test.csv -o scores.csv
```

```python
from HADES_CALCULATOR import HADES_score

custom_hades = HADES_score(models_dir="quick_test_run/models", x_train_parquet="quick_test_run/X_train.parquet")
custom_hades.score(['C1CCCC1', 'c1ccccc1'])
```

## Tests

A small pytest suite guards the two things that have broken silently before: the
scores themselves, and the alignment between molecules and their results.

```bash
pytest
```

It takes about fifteen seconds, and must be run from a checkout: the suite reads
the paper's train/test split, which the installed package does not carry. It
checks:

- three known molecules against recorded scores (`CCO` → 0.0178465), so a change
  to the models, the descriptor set or the featurizer cannot pass unnoticed;
- that featurizing a list containing an unparseable SMILES leaves the surviving
  rows matched to the molecules they came from;
- that a list skips what it cannot parse, and that a single unparseable SMILES
  raises `ValueError`;
- that the `SMILES` column of the training parquet is found by name, and that
  descriptors which do not match the models' features raise instead of scoring;
- that output printed before scoring reaches a pipe exactly once.

One more test trains a model end to end on `data_to_test.csv` and scores with it.
It takes about 16 minutes, so it is skipped by default:

```bash
pytest -m slow
```

The suite loads the ensemble with `verify="require"`, so it also fails if a model
file no longer matches `SHA256SUMS`.

---

## Quiet Setup (`quiet_noise.py`)

Suppresses noisy console output from `chemprop`, `admet_ai`, LightGBM, and `tqdm`
progress bars during model loading and inference.

**Usage:** Import it first, before any other heavy library.

Note that importing it applies two process-wide changes that are never undone:
`warnings` are set to "ignore", and `tqdm` is patched to `disable=True`.

The feature-selection sweep opts back out with `disable=False`
(`HADES/correlation_drop.py`), because it is the slowest step in training and
needs to show progress. It is the one progress bar you will see during a run —
see [How long training takes](#how-long-training-takes) for what to expect from
it.
