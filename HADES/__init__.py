"""HADES -- a soft-voting ensemble that scores molecules from their SMILES.

The public API lives in the two top-level modules shipped alongside this
package, `HADES_CALCULATOR` (scoring) and `train_your_HADES` (training). They
are re-exported here for convenience:

    from HADES import HADES_score, HADESTrainer

The re-export is lazy (PEP 562), and has to be. Both modules import from this
package (`from HADES.quiet_noise import suppress_output`), so importing them
eagerly here would make `import HADES_CALCULATOR` re-enter `HADES/__init__.py`
while `HADES_CALCULATOR` is still half-initialised, and the re-export would
fail with a circular-import error. Deferring to attribute access keeps both
import orders working -- and keeps a bare `import HADES` from pulling in torch,
admet_ai and the gradient-boosting libraries.
"""

__version__ = "1.0.0"
__all__ = ["HADES_score", "HADESTrainer"]

_LAZY_EXPORTS = {"HADES_score": "HADES_CALCULATOR", "HADESTrainer": "train_your_HADES"}


def __getattr__(name):
    """Import a re-exported name from its top-level module on first access."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        return getattr(import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*__all__, "__version__"])