![HADES drug-likeness score logo](https://raw.githubusercontent.com/denovosciencesai/HADES/main/hades_logo.svg)

# HADES — Holistic AI-based Drug-likeness Estimation Score

[![Paper: J. Chem. Inf. Model. 2026](https://img.shields.io/badge/paper-J.%20Chem.%20Inf.%20Model.%202026-1f6feb)](https://doi.org/10.1021/acs.jcim.5c02953)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/denovosciencesai/HADES/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://github.com/denovosciencesai/HADES/blob/main/environment.yml)
[![MCP server](https://img.shields.io/badge/MCP-server-8A2BE2)](#mcp-server-use-hades-from-claude-and-other-ai-assistants)
[![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-D97757)](https://github.com/denovosciencesai/HADES/blob/main/docs/mcp.md#claude-code-plugin)

HADES (Holistic AI-based Drug-likeness Estimation Score) is an open-source Python
package, command-line tool and MCP server that predicts the oral drug-likeness of
small molecules from their SMILES strings. It returns a score from 0 to 1, a
data-driven complement to QED and Lipinski's rule of five, for ranking and
filtering compounds in drug discovery, virtual screening and medicinal chemistry.
The score is the mean predicted probability of a soft-voting ensemble of five
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
[Installation](#installation) ·
[Scoring](#score-molecules) ·
[Training](#train-your-own-model) ·
[MCP and Claude](#mcp-server-use-hades-from-claude-and-other-ai-assistants) ·
[Model integrity](#model-files-and-integrity) ·
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

Each SMILES is standardized (common counter-ions stripped, charges neutralized,
stereochemistry removed) and described by 298 molecular descriptors: Mordred
descriptors, ADMET properties predicted by ADMET-AI, and six of the eight
properties behind QED. Each of the five classifiers predicts the probability that
the molecule belongs with the approved oral drugs, and the HADES score is the mean
of the five.

The shipped models come from the paper's dataset of 8,106 compounds, 1,488
approved oral drugs and 6,618 non-drug molecules, split 80/20 into the training
and test sets in `train_test_saved/`.

## Installation

HADES needs Python 3.10 and is easiest to install with conda (Miniconda or
Anaconda), because RDKit and other compiled dependencies install most reliably
from conda-forge:

```bash
conda env create -f environment.yml
conda activate hades
```

That installs Python 3.10, RDKit and every pinned package in `requirements.txt`.
Optionally, install the project into the environment as well:

```bash
pip install -e .
```

That makes `from HADES import HADES_score, HADESTrainer` (the same classes as
`HADES_CALCULATOR` and `train_your_HADES`) importable from any directory and adds the commands `hades-score` and `hades-train`, which take the
same options as `python HADES_CALCULATOR.py` and `python train_your_HADES.py`, and
`hades-mcp`, the [MCP server](#mcp-server-use-hades-from-claude-and-other-ai-assistants), which also needs the
`[mcp]` extra. The trained models and the descriptor spec are installed alongside
the code, so all three work from any directory.

**Using HADES from Claude or another AI assistant?** After this, follow
[MCP server](#mcp-server-use-hades-from-claude-and-other-ai-assistants) to add
the MCP SDK and connect it.

## Score molecules

In Python, `score()` returns a float for one SMILES and a DataFrame (columns
`SMILES`, `HADES_Score`) for a list:

```python
from HADES_CALCULATOR import HADES_score

hades = HADES_score()
hades.score(["CCCC", "NOT_A_SMILES", "CCO"])
# Could not calculate 1 SMILES: ['NOT_A_SMILES']
#    SMILES  HADES_Score
#      CCCC     0.000458
#       CCO     0.017847
```

A list skips what RDKit cannot parse, scores the rest and says what it skipped,
so the DataFrame can be shorter than the input. A single unparseable SMILES
raises `ValueError` instead. Library log messages are silenced by default;
`HADES_score(quiet=False)` brings them back.

From the command line, score a CSV and get it back with a `HADES_Score` column:

```bash
python HADES_CALCULATOR.py -i data_to_test.csv -o output.csv --smiles-col SMILES
```

Every input row is kept, in order. Rows whose SMILES cannot be parsed, or is
empty, get an empty `HADES_Score`, and the run prints which ones they were.

## Train your own model

Given a CSV of molecules labelled positive and negative, train a new ensemble:

```bash
python train_your_HADES.py -i data_to_test.csv -o my_model \
    --smiles-col SMILES --label-col phase --positive-label 4 --negative-label 0
```

```python
from train_your_HADES import HADESTrainer

HADESTrainer(output_dir="my_model", smiles_col="SMILES", label_col="phase",
             positive_label=4, negative_label=0).run("data_to_test.csv")
```

Expect at least 15 minutes even on a few molecules: feature selection has a large
fixed cost. The run writes its models, data splits and settings into `my_model/`,
and refuses to overwrite an existing folder unless you pass `--overwrite`.
[docs/training.md](https://github.com/denovosciencesai/HADES/blob/main/docs/training.md) has the timings, the output layout,
retraining with `--reuse-hpo` and custom hyperparameter searches.

### Score with a model you trained

Point HADES at the run's models and its training parquet:

```bash
python HADES_CALCULATOR.py --models-dir my_model/models --x-train my_model/X_train.parquet -i data_to_test.csv -o scores.csv
```

```python
custom_hades = HADES_score(models_dir="my_model/models", x_train_parquet="my_model/X_train.parquet")
```

## MCP server: use HADES from Claude and other AI assistants

`hades_mcp.py` is a cheminformatics MCP server: it serves HADES over the
[Model Context Protocol](https://modelcontextprotocol.io), so Claude and other AI assistants can score molecules while they work. It exposes
one tool, `hades_score`, which takes a list of SMILES and returns a score for each,
plus a reason for every input it could not score. Scoring runs on your machine.

Add the MCP SDK to the `hades` environment once:

```bash
conda activate hades
pip install "mcp==2.3.0"
```

Then install the Claude Code plugin, which asks for the path of the environment's
Python (`conda run -n hades python -c "import sys; print(sys.executable)"`):

```
/plugin marketplace add denovosciencesai/HADES
/plugin install hades@denovo-hades
```

Or register the server with Claude Code directly, or with any other MCP client by
the same two paths:

```bash
claude mcp add --scope user hades -- /path/to/envs/hades/bin/python /path/to/HADES/hades_mcp.py
```

Check it with `/mcp`, then ask Claude for the HADES score of a molecule.
[docs/mcp.md](https://github.com/denovosciencesai/HADES/blob/main/docs/mcp.md) covers installing from a shell or a local clone, Claude
Desktop and other clients, Windows and WSL, the tool's output and its configuration.

## Model files and integrity

The pretrained models are Python pickles, and **unpickling a file executes code
contained in it**: only load models from this repository or from a training run
of your own. `HADES_score` checks the shipped models against
`HADES/data/models/SHA256SUMS` before loading them, and refuses to load a file
that does not match. [docs/model-integrity.md](https://github.com/denovosciencesai/HADES/blob/main/docs/model-integrity.md) lists the
checksums and the `verify=` options, and shows how to add a manifest to your own
models.

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

**Which molecules can HADES score (salts, fragments, invalid SMILES)?**
Whole molecules that RDKit can parse. The counter-ions on RDKit's default salt
list (Na+, K+, Cl-, mesylate, tartrate and other common ones) are stripped and
charges neutralized first, so those salt forms score as their parent molecule.
Others, such as citrate, besylate or succinate, stay attached, as does the second
copy of the parent in a 2:1 salt, and they change the score: to score a drug,
pass its parent compound (free acid or base). Empty inputs and fragments with
attachment points (`*`) are rejected rather than scored; the MCP server also
rejects inputs that salt stripping leaves empty, such as sodium chloride or
water. Scores mean most for drug-sized organic molecules like the ones in the
training data; a low score for chemistry far outside it is not evidence that a
molecule is not drug-like.

**How fast is HADES?**
After the models load, about 30 ms per molecule on a laptop CPU (an Intel Core 5
120U: 200 compounds in 5.6 s, 2,000 in 60.4 s), so 10,000 molecules take about
five minutes. Training is much slower; see [docs/training.md](https://github.com/denovosciencesai/HADES/blob/main/docs/training.md).

**Can I use HADES commercially?**
Yes. HADES is released under the MIT license, which allows commercial use; see
[LICENSE](https://github.com/denovosciencesai/HADES/blob/main/LICENSE).

## Contributing

[CONTRIBUTING.md](https://github.com/denovosciencesai/HADES/blob/main/CONTRIBUTING.md) covers setting up, the repository layout,
running the tests and submitting changes. To report a security problem, follow
[SECURITY.md](https://github.com/denovosciencesai/HADES/blob/main/SECURITY.md).

## License

MIT. See [LICENSE](https://github.com/denovosciencesai/HADES/blob/main/LICENSE).
