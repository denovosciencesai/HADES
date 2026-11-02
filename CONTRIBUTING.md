# Contributing to HADES

Thanks for helping. Bug reports, fixes and improvements are all welcome.
To report a security problem, follow [SECURITY.md](SECURITY.md) instead of
opening an issue.

## Setting up

```bash
conda env create -f environment.yml
conda activate hades
pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest` and `ruff`; the environment itself carries only
what HADES needs at runtime. CI installs the same extra, so it runs the checks
you run.

HADES targets Python 3.10 only. The pickled models and the pinned stack were
built on it.

## Repository layout

```
.
├── HADES/                      # the package
│   ├── featurize.py            # Mordred, ADMET-AI and QED descriptors for SMILES
│   ├── correlation_drop.py     # drops highly correlated features (training)
│   ├── hyperparameter_optimization.py, param_spaces.py   # Optuna search and its space
│   ├── integrity.py            # writes and checks the SHA-256 manifest of a model directory
│   ├── quiet_noise.py          # silences library noise; see below
│   └── data/                   # shipped with the package, so an installed HADES finds it anywhere
│       ├── models/             # the five trained models (.pkl) and their SHA256SUMS
│       └── feature_spec.parquet  # descriptor order and means, derived from X_train.parquet
├── HADES_CALCULATOR.py         # HADES_score and the hades-score command
├── train_your_HADES.py         # HADESTrainer and the hades-train command
├── hades_mcp.py                # the MCP server and the hades-mcp command
├── .claude-plugin/             # Claude Code plugin manifest and marketplace
├── parameters_test.py          # a small example hyperparameter space
├── train_test_saved/           # the paper's train/test split (not shipped with the package)
├── data_to_test.csv            # a small sample to score and train on
├── docs/                       # longer guides: MCP, training, model integrity
└── tests/                      # pytest suite
```

`HADES/quiet_noise.py` silences console output from chemprop, admet_ai, LightGBM
and `tqdm`. Importing it changes two things for the whole process, for good:
`warnings` are set to "ignore" and `tqdm` is patched to `disable=True`. The
feature-selection sweep in `HADES/correlation_drop.py` opts back in with
`disable=False`, because it is the slowest step of training and needs to show
progress.

## Running the tests

Run from a checkout -- the suite reads the paper's train/test split, which the
installed package does not carry:

```bash
pytest            # ~15 seconds
pytest -m slow    # trains a model end to end, ~16 minutes
```

Run `pytest -m slow` too if your change touches training (`train_your_HADES.py`,
`HADES/correlation_drop.py`, `HADES/hyperparameter_optimization.py`,
`HADES/param_spaces.py`).

The suite pins three molecules to recorded scores, checks that molecules and
results stay matched when some SMILES cannot be parsed, and loads the ensemble
with `verify="require"`, so it also fails if a model file no longer matches
`SHA256SUMS`.

`tests/test_mcp.py` runs only when the MCP SDK is installed
(`pip install -e ".[mcp]"`); install it if your change touches `hades_mcp.py` or
the plugin. It starts the server in subprocesses, so it adds about two minutes.

## If a change moves the scores

`tests/test_hades.py` pins three molecules to recorded scores (`GOLDEN`). If
your change makes those fail, do not just update the numbers. Say in the PR why
the scores changed and whether that was the intention. Scores that users
already rely on only change on purpose.

If you retrain or replace the shipped models, regenerate their manifest so
verification still passes:

```python
from HADES.integrity import write_manifest

write_manifest("HADES/data/models")
```

## Style

- Indent with 4 spaces. `HADES/featurize.py`, `HADES/correlation_drop.py` and
  `HADES/hyperparameter_optimization.py` still use tabs; match the surrounding
  file until they are converted.
- Match the naming, comment density and idiom of the code around your change.
- Keep a PR to one change. Do not mix whitespace or reformatting with real
  changes.

## What not to commit

- Data files. `*.csv` is gitignored on purpose; only `data_to_test.csv` ships.
- Training runs (`runs/`, `quick_test_run/`, `catboost_info/`, ...).
- Notebooks.

## Pull requests

- Describe what changed and why, and link the issue if there is one.
- Add or update a test when you fix a bug, so it cannot come back unnoticed.
- Make sure `pytest` passes.

## Releasing

Bump the version in `pyproject.toml`, `HADES/__init__.py`, `CITATION.cff` and
`.claude-plugin/plugin.json` together. Claude Code users get a new version of
the plugin only when the version in `plugin.json` changes; until then,
`claude plugin update` keeps them on the copy they have.
