# HADES

HADES scores molecules (from SMILES strings) using an ensemble of five
gradient-boosting / tree models. You can score molecules with the provided
model, or train your own model on your own labelled compounds.

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
├── HADES/                 # core package (Featurizer, CorrDrop, HPOptimizer)
├── hades_calculator.py    # importable scorer (Jupyter/Python)
├── HADES_csv.py           # score a CSV from the command line
├── hades_pipeline.py      # train your own model + auto-generate a scorer
├── HADES_saved_models/    # the provided trained ensemble (.pkl files)
├── train_test_saved/      # feature columns the provided model expects
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 1. Score molecules with the provided model

### In a Jupyter notebook (or any Python session)

```python
from hades_calculator import HADES

hades = HADES()
hades.score(['CCCCSCC', 'c1ccccc1'])
```

`score()` returns:

- a **DataFrame** (columns `SMILES`, `HADES_Score`) when you pass a list of SMILES,
- a single **float** when you pass one SMILES string, e.g. `hades.score("CCO")`.

Progress bars and library log messages are silenced by default. If you ever
need to see them (for debugging), build with `HADES(quiet=False)`.

### From the command line (CSV in, CSV out)

To score a whole file and get a new CSV with the scores added:

```bash
python HADES_csv.py molecules_10.csv scored_molecules.csv --smiles-col SMILES
```

The output CSV is your input file plus a `HADES_Score` column. Use
`--smiles-col` to name the column that holds the SMILES if it is not `SMILES`.

---

## 2. Train your own model

If you have your own dataset of positive- and negative-labelled molecules, you
can train a new HADES ensemble on it.

**Input:** a CSV with a SMILES column and a label column that marks which
compounds are positive and which are negative.

### Command line

```bash
python hades_pipeline.py train my_data.csv -o my_model \
    --smiles-col SMILES \
    --label-col activity \
    --positive-label active \
    --negative-label inactive
```

`-o my_model` names the output folder. If you omit `-o`, you will be prompted
to type a name. Everything for the run is written inside that folder:

```
my_model/
├── config.json            # settings + the exact feature columns
├── param_spaces.json      # the hyper-parameter search space used
├── hyperparameters.json   # best parameters found
├── data/                  # train/test splits
├── models/                # one .pkl per ensemble member
└── score.py               # auto-generated scorer for THIS model
```

Because each run has its own folder, training a new model never overwrites a
previous one.

### In a Jupyter notebook

```python
from hades_pipeline import HADESTrainer

HADESTrainer(
    output_dir="my_model",
    smiles_col="SMILES",
    label_col="activity",
    positive_label="active",
    negative_label="inactive",
).run("my_data.csv")
```

### Customise the hyper-parameter search (optional)

```bash
python hades_pipeline.py dump-config my_space.json   # write the default ranges
# edit the ranges in my_space.json ...
python hades_pipeline.py train my_data.csv -o my_model --param-config my_space.json
```

---

## 3. Use a model you trained

Every trained run folder contains its own `score.py`, which already knows which
models and feature columns to use. Point it at a CSV of molecules:

```bash
python my_model/score.py new_molecules.csv scored.csv --smiles-col SMILES
```

The output CSV is your input plus a `HADES_Score` column. The script finds the
`HADES` package automatically (the path is recorded at training time, with a
fallback that searches parent folders), so you do not need to copy `HADES` into
the run folder — just keep the run folder inside the repo.


write about quiet_noise.py like this
## Quiet Setup (`quiet_setup.py`)

Suppresses noisy console output from `chemprop`, `admet_ai`, LightGBM, and `tqdm`
progress bars during model loading and inference.

**Usage:** Import it first, before any other heavy library:



