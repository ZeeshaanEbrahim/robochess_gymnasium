"""D26 determinism check: 10,000 sampled (start, target) pairs, seed 123, default config.

Run from the repo root:  PYTHONPATH=. python results/d26_excluded_squares/determinism.py
Prints the pair count and a SHA-256 of repr(pairs). The hash must not change
when excluded_squares is left at its default (empty).
"""
import hashlib

from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask

base = RoboChessEnv()
task = T1PickAndPlaceTask(base)
base.reset(seed=123)
pairs = [task._resolve_squares(None, None) for _ in range(10000)]
print(len(pairs), hashlib.sha256(repr(pairs).encode()).hexdigest())
