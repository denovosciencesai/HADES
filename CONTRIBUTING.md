# Contributing to HADES

Thanks for helping. Bug reports, fixes and improvements are all welcome.
To report a security problem, follow [SECURITY.md](SECURITY.md) instead of
opening an issue.

## Setting up

The conda environment is the contributor path. Users do not need it: the models
ship inside the package, so scoring installs with `pip`, and the MCP server runs
through `uvx`. Keep it that way — if a change makes conda necessary to *score*,
it belongs behind a flag or a different design.

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
├── .claude-plugin/             # marketplace manifest (lists the plugin below)
├── plugin/                     # the Claude Code plugin: its manifest, and nothing else
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
CI installs `.[dev,mcp]`, so it always runs there.

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
`plugin/.claude-plugin/plugin.json` together. Claude Code users get a new version of
the plugin only when the version in `plugin.json` changes; until then,
`claude plugin update` keeps them on the copy they have.

`deploy/Dockerfile` pins the release it serves as well, in `ARG
HADES_VERSION`. Bump it in the same commit: left behind, the next image built
from it serves the previous release, and because the container reports its own
version in `serverInfo` the mismatch is visible to every client but to nobody
reading the repository.

`plugin.json` carries the version **twice**: its own `version`, and the
`hades-druglikeness[mcp]==<version>` pin in the `uvx` arguments that decides
which release the plugin actually runs.

**Merge the plugin pin to `main` only once that version is on PyPI.** The
marketplace serves from the default branch, so between the merge and the
upload every `/plugin install` and `claude plugin update` produces a server
that fails to start, with a uv resolution error rather than anything about
HADES. 0.9.1 was released this way and had exactly that window. Either publish
the release from the tag before merging the bump, or keep the pin in a
follow-up pull request and merge it after the upload.

The same pin appears in the plugin's warm-up command in `README.md` — twice,
once for bash and once for PowerShell — and again in `docs/mcp.md`. It has to:
uv caches one environment per resolution, so a warm-up whose pin, extra index
or index strategy differs from the plugin's fills a different cache entry and
leaves the plugin installing from scratch on first start — the 15-minute hang
the warm-up exists to prevent. If you change the `uvx` arguments or the `env`
block in `plugin.json`, change those commands to match in the same commit.

Nine places in all, and it is worth listing them rather than counting:

1. `pyproject.toml`
2. `HADES/__init__.py`
3. `CITATION.cff`
4. `plugin/.claude-plugin/plugin.json`, its own `version`
5. `plugin/.claude-plugin/plugin.json`, the `uvx` pin
6. `README.md`, the bash warm-up
7. `README.md`, the PowerShell warm-up
8. `docs/mcp.md`, the warm-up
9. `deploy/Dockerfile`, `ARG HADES_VERSION`

The image tags in `deploy/Dockerfile`'s header comment and in
`deploy/README.md` are examples rather than pins, but they age badly if left
behind. `grep -rn "<previous version>"` over tracked files before tagging
catches all of it, and has already caught a miss that would have shipped.

The package is published to PyPI as `hades-druglikeness` by
`.github/workflows/publish.yml`, through PyPI trusted publishing (no API token):

1. Merge the version bump into `main`.
2. Optional trial: Actions → publish → Run workflow uploads to TestPyPI. Check
   it installs with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hades-druglikeness`,
   or run Actions → install-check (index `testpypi`), which installs it on
   Linux, Windows and an Apple Silicon Mac, scores a molecule and calls the
   MCP tool. Run it again with index `pypi` after the release.
3. Create a GitHub release with tag `v<version>`, for example `v1.0.1`.
   Publishing the release uploads to PyPI. The workflow stops if the tag does
   not match the version in `pyproject.toml`.

A version number can be uploaded to PyPI only once, even after it is deleted,
so a mistake means a new version.

The release is not finished at the upload. The hosted server installs from
PyPI too, so until its image is rebuilt it runs the previous release and
reports that version in `serverInfo`:

4. Rebuild the image at the new `ARG HADES_VERSION` and update the container
   app (see [deploy/README.md](deploy/README.md)).
5. Call the endpoint and check `serverInfo` reports the version you just
   released.
