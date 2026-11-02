# Training a HADES model

The [README](../README.md#train-your-own-model) has the commands to train an
ensemble and score with it. This page covers the details: how long it takes, what
a run writes, retraining, and a custom hyperparameter search.

## How long training takes

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
**15 minutes as a floor you pay regardless of dataset size**. Its progress bar is
the only one you will see during a run.

**Step 5/6, hyperparameter search**, is the stage that actually scales — under 2
minutes at 20 compounds, 12 minutes at 2000, and still climbing. Above a few
thousand compounds, expect it to overtake feature selection.

Featurizing (step 2/6) grows linearly with compound count and stays modest.

Scoring is cheap next to training: about 30 ms per molecule (200 compounds in
5.6s, 2000 in 60.4s), so 10,000 molecules take roughly five minutes.

## What a run writes

`-o output_dir` (`output_dir=` in Python) names the output folder. Everything for
the run is written inside it:

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

Molecules whose SMILES cannot be parsed are dropped before training, and step
2/6 reports how many. Labels stay matched to the molecules that survived.

## Retraining into the same folder

Because each run has its own folder, training a new model never overwrites a
previous one: if the folder already exists, training stops with an error. Pass
`--overwrite` (`overwrite=True` in Python) to train into an existing folder on
purpose. That replaces the models in it -- any `.pkl` files an earlier run left in
`models/` are deleted first, so the two runs are never mixed.

To retrain without repeating the hyperparameter search, pass `--reuse-hpo`
together with `--overwrite`. Step 5/6 then loads `hyperparameters.json` from the
earlier run in that folder instead of searching again.

## Customize the hyperparameter search

To search a different parameter space, write a `.py` file structured like
`HADES/param_spaces.py` and pass it in. `parameters_test.py` is a small,
ready-made example:

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

Training does not write a checksum manifest for its models; see
[Your own trained models](model-integrity.md#your-own-trained-models) to add one.
