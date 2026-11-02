"""Put the repository root on `sys.path` for the test suite.

`test_training.py` imports `parameters_test`, a repository-only module that is
not part of the installed package. Without this, finding it would depend on
pytest being started from the repository root -- the same cwd dependence the
rest of the suite no longer has.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
