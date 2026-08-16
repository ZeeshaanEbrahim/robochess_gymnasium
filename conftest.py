"""Makes the repository root importable so tests can `import robochess`
without the package being pip-installed."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
