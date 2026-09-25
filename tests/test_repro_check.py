"""F13: scripts/repro_check.py must build the algorithm the config names.

It used to construct PPO unconditionally, so `--config <a SAC config>` recorded
a PPO trace and the reproducibility gate said nothing about SAC (or HER).

Training runs in a SUBPROCESS, not in the pytest process. In this project's
local environment, rendering a frame (GLFW or OSMesa, which tests/test_env_basic.py
does) and THEN importing torch._dynamo / triton -- which SB3's first learn() call
does lazily -- segfaults the interpreter (reproduced outside pytest; the reverse
order is fine). A fresh process has not rendered anything, so this test does not
depend on which tests ran before it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Runs repro_check.run twice on one config, spying on which algorithm's learn()
# executes, then compares the two traces exactly. Prints one JSON line.
_DRIVER = r"""
import importlib.util, json, sys, tempfile, warnings
from pathlib import Path
import numpy as np
warnings.simplefilter("ignore")
repo, config = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("repro_check", repo / "scripts" / "repro_check.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
from robochess.experiment import load_config, resolve_config
from stable_baselines3 import PPO, SAC
cfg = resolve_config(load_config(str(repo / "configs" / config)), config_path=repo / "configs" / config)

learned = []
for cls in (PPO, SAC):
    def make(orig):
        def spy(self, *a, **k):
            learned.append(type(self).__name__)
            return orig(self, *a, **k)
        return spy
    cls.learn = make(cls.learn)

out = Path(tempfile.mkdtemp())
rc.run(cfg, 150, "A", out)      # > SAC's 100 learning_starts
rc.run(cfg, 150, "B", out)
A, B = dict(np.load(out / "trace_A.npz")), dict(np.load(out / "trace_B.npz"))
identical = A.keys() == B.keys() and all(np.array_equal(A[k], B[k]) for k in A)
print(json.dumps(dict(learned=learned, identical=bool(identical))))
"""


@pytest.mark.parametrize("config, expected", [
    ("t1_sac_shaped_state_seed0.yaml", "SAC"),
    ("t1_sac_sparse_state_seed0.yaml", "SAC"),   # SAC + HER replay buffer
    ("t1_ppo_shaped_state_seed0.yaml", "PPO"),
])
def test_run_trains_the_algorithm_the_config_names(config, expected):
    proc = subprocess.run([sys.executable, "-c", _DRIVER, str(REPO), config],
                          capture_output=True, text=True, timeout=300, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["learned"] == [expected, expected]      # run A and run B
    assert result["identical"], "two same-seed runs diverged"
