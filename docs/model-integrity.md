# Model files and integrity

The five pretrained models ship as Python pickles (`HADES/data/models/*.pkl`).
**Unpickling a file executes code contained in it.** Only load model files from a
source you trust — this repository, or a training run you produced yourself. That
applies in particular when you point `--models-dir` at a directory of models: if
the directory came from somewhere else, you are running whatever is in it.

## Published checksums

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

## Checking on load

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

## Your own trained models

Training does not write a manifest, so scoring with a model you trained yourself
prints a warning that the directory is unverified. To record its digests once:

```python
from HADES.integrity import write_manifest

write_manifest("quick_test_run/models")
```

After that the directory verifies like the shipped one.

## What this does and does not buy you

Checksums detect tampering and corruption, and they pin which bytes produced a
given set of scores. They do **not** make an untrusted pickle safe: a hostile file
whose digest matches the manifest still executes on load. Verification is only
meaningful if the manifest itself reached you through a channel you trust.
