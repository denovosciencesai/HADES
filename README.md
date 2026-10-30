# HADES — Holistic AI-based Drug-likeness Estimation Score

[![Paper: J. Chem. Inf. Model. 2026](https://img.shields.io/badge/paper-J.%20Chem.%20Inf.%20Model.%202026-1f6feb)](https://doi.org/10.1021/acs.jcim.5c02953)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](environment.yml)
[![MCP server](https://img.shields.io/badge/MCP-server-8A2BE2)](#4-use-hades-from-claude-and-other-mcp-clients)
[![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-D97757)](#claude-code-plugin)

HADES (Holistic AI-based Drug-likeness Estimation Score) predicts the oral
drug-likeness of small molecules from their SMILES strings. It returns a score
between 0 and 1: the mean predicted probability of a soft-voting ensemble of five
machine-learning classifiers (HistGradientBoosting, RandomForest, CatBoost,
XGBoost and LightGBM) trained on Mordred, ADMET-AI and QED descriptors.

This repository is the authors' implementation of
[Petrosyan et al., *J. Chem. Inf. Model.* 2026, 66 (16), 9944–9960](https://doi.org/10.1021/acs.jcim.5c02953).
It ships the paper's pretrained models and lets you:

- score molecules from Python or the command line (CSV in, CSV out);
- score them from Claude Code, Claude Desktop or any other MCP client, through the
  bundled MCP server and Claude Code plugin;
- train a new HADES ensemble on your own labelled compounds.

**Contents:** [Quick start](#quick-start) ·
[Setup](#setup) ·
[Scoring](#1-score-molecules-with-the-provided-model) ·
[Training](#2-train-your-own-model) ·
[MCP server and Claude Code plugin](#4-use-hades-from-claude-and-other-mcp-clients) ·
[Model integrity](#model-artifacts-and-integrity) ·
[Tests](#tests) ·
[How to cite](#how-to-cite) ·
[FAQ](#faq)

## Quick start

```bash
git clone https://github.com/denovosciencesai/HADES.git
cd HADES
conda env create -f environment.yml
conda activate hades
python HADES_CALCULATOR.py -i data_to_test.csv -o scores.csv
```

`scores.csv` is `data_to_test.csv` with a `HADES_Score` column added. From Python:

```python
from HADES_CALCULATOR import HADES_score

hades = HADES_score()
hades.score("CCO")                # 0.0178...
hades.score(["CCO", "c1ccccc1"])  # DataFrame with SMILES and HADES_Score columns
```

### How it works

Each SMILES is standardized (salts stripped, charges neutralized, stereochemistry
removed) and described by 298 molecular descriptors: Mordred descriptors, ADMET
properties predicted by ADMET-AI, and six of the eight properties behind QED.
Each of the five classifiers predicts the probability that the molecule belongs
with the approved oral drugs, and the HADES score is the mean of the five.

The shipped models come from the paper's dataset of 8,106 compounds, 1,488
approved oral drugs and 6,618 non-drug molecules, split 80/20 into the training
and test sets in `train_test_saved/`.

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
default model and training-data paths are relative to the current directory, so
run them from the repository root, or pass `--models-dir` and `--x-train`.
`pip install -e ".[mcp]"` adds a third command, `hades-mcp`, the MCP server of
[section 4](#4-use-hades-from-claude-and-other-mcp-clients); it finds the
repository's models from any directory and is configured through environment
variables instead.

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
          └──quiet_noise.py                     # silences warnings, tqdm bars, etc. for a cleaner console; can be disabled for debugging
├── HADES_saved_models/                         # the trained ML models (.pkl files) for HADES_score
          └──SHA256SUMS                         # SHA-256 digest of each model file, checked before loading
├── train_test_saved/                           # split data used in the original paper
├── HADES_CALCULATOR.py                          # script to calculate HADES_score from the terminal or a Jupyter notebook, for single molecules or batches
├── train_your_HADES.py                          # script to train your own HADES model on your own data
├── parameters_test.py                           # a small hyperparameter set for testing custom parameter spaces during HADES training
├── hades_mcp.py                                # MCP server: HADES as a tool for Claude and other AI assistants
├── .claude-plugin/                             # Claude Code plugin manifest and plugin marketplace for this repository
├── tests/                                      # pytest suite: golden scores and regression checks
├── environment.yml                             # defines the HADES conda environment
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

The five pretrained models ship as Python pickles (`HADES_saved_models/*.pkl`).
**Unpickling a file executes code contained in it.** Only load model files from a
source you trust — this repository, or a training run you produced yourself. That
applies in particular to section 3 below, where you point `--models-dir` at a
directory of models: if the directory came from somewhere else, you are running
whatever is in it.

### Published checksums

`HADES_saved_models/SHA256SUMS` records the SHA-256 digest of every shipped model:

```
e990b6cdd5c83fbc6c186b23df325c6a10d0b6969261f5dc9a7376f493a6e3ec  CatBoostClassifier.pkl
b8946b76986e65372135d152e69f088338cfba66a792c441b1b2da246e2afd19  HistGradientBoostingClassifier.pkl
c96a8d48d24e2ac6e58531ff6d8dc38d85058acc798ce90dceddb24bd0866dbf  LGBMClassifier.pkl
2d5fbebafa31241202d59cc3e1f5e82f79b1d07453cfc0c2e421d8f31fa26e83  RandomForestClassifier.pkl
10314d71ccf26a22df20d646bb8837d6cd163dab22d0cafd7c8d95c418bd877d  XGBClassifier.pkl
```

It is a standard `sha256sum` file, so it can be checked without Python:

```bash
cd HADES_saved_models && sha256sum -c SHA256SUMS
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

---

## 4. Use HADES from Claude and other MCP clients

`hades_mcp.py` serves HADES over the [Model Context Protocol](https://modelcontextprotocol.io)
(MCP), so an AI assistant can score molecules while it works: ask Claude to rank a
set of analogues by HADES score and it calls HADES directly. The server exposes one
tool, `hades_score`, which takes a list of SMILES and returns a score for each
molecule, plus a reason for every input it could not score.

The server runs on your machine, in the `hades` environment, which needs the MCP
SDK added once:

```bash
conda activate hades
pip install "mcp==2.3.0"
```

From a checkout, `pip install -e ".[mcp]"` does the same and also installs HADES
itself, with a `hades-mcp` command; it re-applies every pin in `requirements.txt`
too. The environment works on native Windows as well as Linux (macOS is
untested). Install it on the system that runs your MCP client: a Windows client
cannot use the plugin with an environment that lives in WSL (see
[Other MCP clients](#other-mcp-clients)).

### Claude Code plugin

This repository is a [Claude Code plugin](https://code.claude.com/docs/en/plugins)
and its own plugin marketplace, `denovo-hades`. In Claude Code 2.1.275 or later:

```
/plugin install hades --marketplace denovosciencesai/HADES
```

On older versions, add the marketplace first (`/plugin marketplace add denovosciencesai/HADES`),
then run `/plugin install hades@denovo-hades`. To install from a local clone
instead of GitHub, add the marketplace by path: `/plugin marketplace add /path/to/HADES`.

The plugin asks for one setting: the absolute path of the `hades` environment's
Python interpreter. To print it:

```bash
conda run -n hades python -c "import sys; print(sys.executable)"
```

From a shell, the same install is:

```bash
claude plugin marketplace add denovosciencesai/HADES
claude plugin install hades@denovo-hades --config python=/home/you/miniconda3/envs/hades/bin/python
```

A shell install never prompts, so pass the interpreter with `--config`, or set it
later with `/plugin configure hades@denovo-hades`. Write the path out in full:
`~` is not expanded, and a wrong path only shows up when the server fails to
start. On Windows the path looks like
`C:/Users/you/miniconda3/envs/hades/python.exe`. Check the server with `/mcp`;
the tool appears as `mcp__plugin_hades_hades__hades_score`. The plugin was tested
with Claude Code 2.1.299 on Windows.

A GitHub install keeps two copies of the repository, models included: the
marketplace clone and the plugin cache, about 210 MB together. Installing from a
local clone copies that whole directory, untracked files included, so install
from a clean checkout. The plugin is pinned to a version, and third-party
marketplaces do not auto-update by default, so pull a new release with
`claude plugin update hades@denovo-hades`.

The plugin runs one thing: `hades_mcp.py`, with the interpreter you configured,
on your machine. Scoring is local. Before loading the pickled models, the server
checks them against `SHA256SUMS`, as described in
[Model artifacts and integrity](#model-artifacts-and-integrity).

To work on the plugin itself, start Claude Code with `claude --plugin-dir .` from
a checkout and run `/plugin configure hades` once. After editing `hades_mcp.py` or
the `HADES` package, reconnect the server in `/mcp`, or restart Claude Code, so a
new server process picks up the change.

### Other MCP clients

Every MCP client needs the same two things: the `hades` environment's Python and
the path to `hades_mcp.py`. In Claude Code without the plugin:

```bash
claude mcp add --scope user hades -- /home/you/miniconda3/envs/hades/bin/python /path/to/HADES/hades_mcp.py
```

`--scope user` makes the tool available in every project, as the plugin does;
without it, it exists only in the directory where you ran the command.

In Claude Desktop (`claude_desktop_config.json`), and in clients that use the
same format:

```json
{
  "mcpServers": {
    "hades": {
      "command": "/home/you/miniconda3/envs/hades/bin/python",
      "args": ["/path/to/HADES/hades_mcp.py"]
    }
  }
}
```

On Windows, use the environment's `python.exe` and Windows paths, for example
`C:/Users/you/miniconda3/envs/hades/python.exe` and `C:/path/to/HADES/hades_mcp.py`.

If your client runs on Windows but the `hades` environment lives in WSL, launch
the server through WSL: `"command": "wsl.exe"` with
`"args": ["-d", "Ubuntu", "-e", "/home/you/miniconda3/envs/hades/bin/python", "/home/you/HADES/hades_mcp.py"]`.
The plugin cannot do this, so with Claude Code use `claude mcp add` the same way
(`claude mcp add --scope user hades -- wsl.exe -d Ubuntu -e ...`), or run Claude
Code inside WSL.

### What the tool returns

```json
{
  "results": [{"index": 0, "smiles": "CCO", "hades_score": 0.0178}],
  "invalid": [{"index": 1, "smiles": "NOT_A_SMILES", "reason": "RDKit cannot parse it"}]
}
```

`index` is each entry's position in the input list. A call takes up to 100 SMILES,
which keeps a full result well inside the output limits MCP clients apply. The
models load on the first call, which takes a while; later calls are fast.

### Configuration

The server reads these environment variables, all optional. Where to set them
depends on the client:

- **Claude Code**, plugin or `claude mcp add`: the server inherits the environment
  Claude Code was started from, so export them first. With `claude mcp add` you
  can also pass them with `-e NAME=value`.
- **Claude Desktop and most other clients** pass only a few variables to the
  servers they start. Add an `env` block to the server entry:
  `"env": {"HADES_MODELS_DIR": "/abs/path/run/models", "HADES_X_TRAIN": "/abs/path/run/X_train.parquet"}`.
- **Through `wsl.exe`**, Windows variables reach WSL only when listed in
  `WSLENV`, for example `WSLENV=HADES_MODELS_DIR/p:HADES_X_TRAIN/p`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HADES_MODELS_DIR` | the shipped models | Score with an ensemble you trained ([section 3](#3-use-a-model-you-trained)) |
| `HADES_X_TRAIN` | the one that goes with the shipped models | The `X_train.parquet` that goes with `HADES_MODELS_DIR` |
| `HADES_VERIFY` | `auto` | Checksum check before loading the models: `auto`, `require` or `off` |
| `HADES_MAX_BATCH` | `100` | Maximum SMILES per call |
| `HADES_REPO` | the directory of `hades_mcp.py` | The HADES checkout to import HADES and its shipped models from |

To check the server without any client, score a few molecules directly:

```bash
python hades_mcp.py --selftest CCO c1ccccc1
```

## Tests

A small pytest suite guards the two things that have broken silently before: the
scores themselves, and the alignment between molecules and their results.

```bash
pytest
```

It takes about fifteen seconds, and must be run from the repository root so the
default model and training-data paths resolve. It checks:

- three known molecules against recorded scores (`CCO` → 0.0178465), so a change
  to the models, the descriptor set or the featurizer cannot pass unnoticed;
- that featurizing a list containing an unparseable SMILES leaves the surviving
  rows matched to the molecules they came from;
- that a list skips what it cannot parse, and that a single unparseable SMILES
  raises `ValueError`;
- that the `SMILES` column of the training parquet is found by name, and that
  descriptors which do not match the models' features raise instead of scoring;
- that output printed before scoring reaches a pipe exactly once;
- with the MCP SDK installed, that the MCP server keeps each score matched to its
  input, that real client sessions work over stdio, and that output printed
  during a call reaches stderr instead of the JSON-RPC stream (`tests/test_mcp.py`,
  skipped without the SDK; it starts the server in subprocesses, so it adds about
  a minute).

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

---

## How to cite

If you use HADES, please cite the paper:

> Petrosyan, N.; Matevosyan, H.; Tirosyan, I.; Gabrielyan, Y.; Gevorgyan, S.;
> Harutyunyan, V.; Matevosyan, M.; Zakaryan, H. Machine Learning-Based Soft Voting
> Ensemble Model for the Prediction of Oral Drug-Likeness of Chemical Structures.
> *J. Chem. Inf. Model.* **2026**, *66* (16), 9944–9960.
> <https://doi.org/10.1021/acs.jcim.5c02953>

```bibtex
@article{petrosyan2026hades,
  title   = {Machine Learning-Based Soft Voting Ensemble Model for the Prediction of Oral Drug-Likeness of Chemical Structures},
  author  = {Petrosyan, Narek and Matevosyan, Hovhannes and Tirosyan, Irina and Gabrielyan, Yeva and Gevorgyan, Smbat and Harutyunyan, Vardan and Matevosyan, Mher and Zakaryan, Hovakim},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {66},
  number  = {16},
  pages   = {9944--9960},
  year    = {2026},
  doi     = {10.1021/acs.jcim.5c02953}
}
```

GitHub's **Cite this repository** button, generated from `CITATION.cff`, gives the
same reference in APA and BibTeX.

## FAQ

**What does a HADES score mean?**
It is a number between 0 and 1: the mean probability, across the five models,
that a molecule belongs with the approved oral drugs HADES was trained on. Higher
means more similar to approved oral drugs. Use it to rank and prioritize
compounds; it is not a probability that a compound will succeed in the clinic.

**How is HADES different from QED or Lipinski's rule of five?**
The rule of five and QED judge a molecule from a few physicochemical properties,
through fixed cutoffs or desirability functions. HADES is trained on data: it uses
most of QED's properties together with Mordred descriptors and ADMET-AI
predictions, in models fitted to separate approved oral drugs from non-drug
molecules. See the paper for how it compares with other drug-likeness scores.

**Which molecules can it score?**
Whole molecules that RDKit can parse. Salts and counter-ions are stripped and
charges neutralized first, so a salt form scores as its parent molecule. Empty
inputs and fragments with attachment points (`*`) are rejected rather than scored;
the MCP server also rejects inputs that are nothing but salts, such as sodium
chloride. Scores mean most for drug-sized organic molecules like the ones in the
training data; a low score for chemistry far outside it is not evidence that a
molecule is not drug-like.

**How fast is it?**
After the models load, about 30 ms per molecule on a laptop CPU (an Intel Core 5
120U: 200 compounds in 5.6 s, 2,000 in 60.4 s), so 10,000 molecules take about
five minutes.

**Can I train HADES on my own data?**
Yes. See [Train your own model](#2-train-your-own-model), then
[Use a model you trained](#3-use-a-model-you-trained).

**Can an AI assistant use HADES?**
Yes, through the MCP server. See
[Use HADES from Claude and other MCP clients](#4-use-hades-from-claude-and-other-mcp-clients).

## License

MIT. See [LICENSE](LICENSE).
