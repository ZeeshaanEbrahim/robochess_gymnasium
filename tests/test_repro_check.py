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
has_digest = "final_state_sha256" in A and A["final_state_sha256"].shape == (32,) and A["final_state_sha256"].dtype == np.uint8
print(json.dumps(dict(learned=learned, identical=bool(identical), has_digest=bool(has_digest))))
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
    assert result["has_digest"], "trace carries no final-weights digest"


def test_state_digest_tracks_weights_names_and_log_ent_coef():
    """state_digest hashes the weights (not just shapes), tensor names, and SAC's log_ent_coef."""
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("repro_check", REPO / "scripts" / "repro_check.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)

    class Fake:
        def __init__(self, w=0.0, name="lin", ent=None):
            self.policy = torch.nn.Sequential()
            self.policy.add_module(name, torch.nn.Linear(2, 2))
            with torch.no_grad():
                getattr(self.policy, name).weight.fill_(w)
                getattr(self.policy, name).bias.zero_()
            if ent is not None:
                self.log_ent_coef = torch.tensor([ent])

    base = rc.state_digest(Fake())
    assert base == rc.state_digest(Fake()) and len(base) == 64
    assert base != rc.state_digest(Fake(w=1e-7))                  # a single weight change
    assert base != rc.state_digest(Fake(name="other"))            # a renamed tensor
    assert rc.state_digest(Fake(ent=0.0)) != base                 # SAC's entropy coefficient is covered
    assert rc.state_digest(Fake(ent=0.0)) != rc.state_digest(Fake(ent=1e-7))
