# Contributing to HADES

Thanks for helping. Bug reports, fixes and improvements are all welcome.
To report a security problem, follow [SECURITY.md](SECURITY.md) instead of
opening an issue.

## Setting up

```bash
conda env create -f environment.yml
conda activate hades
pip install -e .
```

HADES targets Python 3.10 only. The pickled models and the pinned stack were
built on it.

## Running the tests

Run from the repository root, so the default model and data paths resolve:

```bash
pytest            # ~15 seconds
pytest -m slow    # trains a model end to end, ~16 minutes
```

Run `pytest -m slow` too if your change touches training (`train_your_HADES.py`,
`HADES/correlation_drop.py`, `HADES/hyperparameter_optimization.py`,
`HADES/param_spaces.py`).

`tests/test_mcp.py` runs only when the MCP SDK is installed
(`pip install -e ".[mcp]"`); install it if your change touches `hades_mcp.py` or
the plugin. It starts the server in subprocesses, so it adds about a minute.

## If a change moves the scores

`tests/test_hades.py` pins three molecules to recorded scores (`GOLDEN`). If
your change makes those fail, do not just update the numbers. Say in the PR why
the scores changed and whether that was the intention. Scores that users
already rely on only change on purpose.

If you retrain or replace the shipped models, regenerate their manifest so
verification still passes:

```python
from HADES.integrity import write_manifest

write_manifest("HADES_saved_models")
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
