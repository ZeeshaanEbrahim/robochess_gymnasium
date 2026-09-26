"""D28 flag-off identity: dump every reward, observation and info of three replays
plus the resolved config of every configs/*.yaml, under whichever tree ROOT names.

    python flag_off_identity.py ROOT OUT.json

ROOT is put first on sys.path, so the code under test is ROOT's own. The output is
compared byte for byte between 76e116c and the place-reward branch. Floats are
written with repr() and observations as sha256 of their bytes, so equal files mean
bit-identical values.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "tests"), str(root / "scripts")]
import numpy as np
import robochess
assert Path(robochess.__file__).resolve().is_relative_to(root), robochess.__file__
import d27_scenarios as sc
import scripted_expert as se
import yaml
from robochess.experiment import resolve_config

spec = importlib.util.spec_from_file_location("train", root / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

BASE = dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state", seed=0,
            total_timesteps=1000, start_square="e4", target_square="e5")


def norm(v):
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() + str(v.shape)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


rows = []


def rec(tag, out):
    obs, r, term, trunc, info = out
    rows.append(dict(tag=tag, reward=repr(float(r)), term=bool(term), trunc=bool(trunc),
                     obs=norm(obs), info=norm(info)))


def env_for(**over):
    return train.build_env(resolve_config(dict(BASE, **over), config_path="t.yaml"))


# 1. joint (absolute-action) expert replay, default shaped reward
for start, target in (("e4", "e5"), ("c1", "f6")):
    env = env_for(action_mode="absolute", max_episode_steps=350, start_square=start,
                  target_square=target)
    orig = env.step
    env.step = lambda a, orig=orig, tag=f"joint {start}->{target}": (lambda o: (rec(tag, o), o)[1])(orig(a))
    se.ScriptedExpert(env).run_episode(seed=0)
    env.close()


# 2. Cartesian expert replays through build_env: a clean place and a push
class BuiltDriver(sc.Driver):
    def __init__(self, start, target, cfgd, budget=300, seed=0):
        self.env = train.build_env(resolve_config(
            dict(BASE, action_mode="cartesian", max_episode_steps=budget,
                 start_square=start, target_square=target, **cfgd), config_path="t.yaml"))
        self.base = self.env.unwrapped
        self.on_step = None
        self.env.reset(seed=seed, options={"start_square": start, "target_square": target})
        b = self.base
        self.piece_xy = b._achieved_goal()[:2].astype(float).copy()
        self.goal_xy = b._desired_goal[:2].astype(float).copy()
        self.trace = sc.Trace()
        self._span = b._ctrl_high[6] - b._ctrl_low[6]
        self._grip_action = {"open": -1.0, "closed": 1.0}


for tag, cfgd in (("cart shaped", {}), ("cart pickup", dict(pickup_mode=True, pickup_obs=True))):
    dr = BuiltDriver("c1", "f6", cfgd)
    dr.on_step = lambda o, tag=tag: rec(tag + " place", o)
    dr.grasp_and_lift(); dr.place_and_release(height_m=0.0015)
    dr.base.close()
    dr = BuiltDriver("e4", "e5", cfgd)
    dr.on_step = lambda o, tag=tag: rec(tag + " push", o)
    sc.push_then_open(dr)
    dr.base.close()

# 3. resolved config of every existing config file
configs = {}
for p in sorted((root / "configs").glob("*.yaml")):
    raw = yaml.safe_load(p.read_text())
    configs[p.name] = norm(resolve_config(raw, config_path=p))

json.dump(dict(rows=rows), open(sys.argv[2], "w"), indent=0, sort_keys=True)
json.dump(configs, open(sys.argv[2] + ".configs", "w"), indent=0, sort_keys=True)
print(robochess.__file__, "steps:", len(rows), "configs:", len(configs))
