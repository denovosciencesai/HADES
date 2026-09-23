# HADES

HADES is a soft-voting ensemble of five ML models (HistGradientBoosting, RandomForest, CatBoost, XGBoost, and LightGBM) that returns a score for a given SMILES string.

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

## Repository layout

```
.
├── HADES ┐                                     # core package
          ├──__init__.py                        # package marker; re-exports HADES_score and HADESTrainer
          ├──correlation_drop.py                # drops highly correlated features
          ├──featurize.py                       # featurizes given SMILES with Mordred, ADMET and QED features
          ├──hyperparameter_optimization.py     # finds best hyperparameters using the Optuna algorithm (https://github.com/optuna/optuna)
          ├──param_spaces.py                    # hyperparameter search space for Optuna
          └──quiet_noise.py                     # silences warnings, tqdm bars, etc. for a cleaner console; can be disabled for debugging
├── HADES_saved_models/                         # the trained ML models (.pkl files) for HADES_score
├── train_test_saved/                           # split data used in the original paper
├── HADES_CALCULATOR.py                          # script to calculate HADES_score from the terminal or a Jupyter notebook, for single molecules or batches
├── train_your_HADES.py                          # script to train your own HADES model on your own data
├── parameters_test.py                           # a small hyperparameter set for testing custom parameter spaces during HADES training
├── environment.yml                             # defines the HADES conda environment
├── pyproject.toml                              # packaging metadata (`pip install -e .`)
├── requirements.txt                            # required libraries, packages, etc.
├── data_to_test.csv                            # small sample dataframe for testing the project
└── README.md                                   # description of the project and its usage
```

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
  # ValueError: Not a valid SMILES: 'NOT_A_SMILES'
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

Every input row is kept. Rows whose SMILES could not be parsed get an empty
`HADES_Score`, and the run prints which ones they were.

---

## 2. Train your own model

If you have your own dataset of positive- and negative-labelled molecules, you
can train a new HADES ensemble on it.

**Input:** a CSV with a SMILES column and a label column marking which
compounds are positive and which are negative.

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
previous one.

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

## Quiet Setup (`quiet_noise.py`)

Suppresses noisy console output from `chemprop`, `admet_ai`, LightGBM, and `tqdm`
progress bars during model loading and inference.

**Usage:** Import it first, before any other heavy library.

Note that importing it applies two process-wide changes that are never undone:
`warnings` are set to "ignore", and `tqdm` is patched to `disable=True`. This
also silences the progress bar of the feature-selection step during training,
which is why `train_your_HADES.py` can run for many minutes with no output.