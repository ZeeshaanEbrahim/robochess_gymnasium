"""D28 audit A2: 20 stochastic episodes with a freshly initialised PPO (with VecNormalize) and SAC
policy on the place config (e4->e5, Cartesian, home start, 300 steps)."""
import collections, importlib.util, json, sys, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(REPO)]
warnings.simplefilter("ignore")
import numpy as np
from robochess.experiment import resolve_config
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
out = {}
for algo in ("ppo", "sac"):
    extra = dict(vecnormalize_clip_reward=1000) if algo == "ppo" else {}
    c = resolve_config(dict(task="t1", algorithm=algo, reward_type="shaped", observation="state", seed=0,
        total_timesteps=1, action_mode="cartesian", max_episode_steps=300, start_square="e4", target_square="e5",
        pickup_obs=True, place_obs=True, place_mode=True, device="cpu", **extra), config_path="t.yaml")
    venv = train.build_vec_env(c); model = train.build_model(c, venv)
    base = venv.venv.envs[0].unwrapped if algo == "ppo" else venv.envs[0].unwrapped
    venv.seed(20000); obs = venv.reset()
    occ = collections.Counter(); q = []; invalid = steps = 0; reasons = collections.Counter(); eps = 0
    reached = collections.Counter(); ep_states = set(); ep_len = []; n = 0
    while eps < 20:
        act, _ = model.predict(obs, deterministic=False)
        obs, r, done, infos = venv.step(act); i = infos[0]; steps += 1; n += 1
        occ[i["place_state"]] += 1; ep_states.add(i["place_state"]); invalid += bool(i["invalid_transition"])
        if not done[0]: q.append(float(base.data.qpos[base._grip_qadr]))
        if done[0]:
            eps += 1; reasons[i.get("terminated_reason") or "time_limit"] += 1; ep_len.append(n); n = 0
            for s in ep_states: reached[s] += 1
            ep_states = set()
    q = np.array(q)
    res = dict(episodes=eps, steps=steps, occupancy={k: round(v / steps, 4) for k, v in sorted(occ.items())},
               gripper_qpos=dict(min=float(q.min()), p10=float(np.percentile(q, 10)), median=float(np.median(q)),
                                 p90=float(np.percentile(q, 90)), max=float(q.max()),
                                 frac_open_below_0_2=float((q < 0.2).mean()), frac_ge_0_6=float((q >= 0.6).mean())),
               invalid_rate=invalid / steps, invalid_steps=invalid, termination_reasons=dict(reasons),
               episodes_reaching_state=dict(reached), episode_lengths=sorted(set(ep_len)))
    out[algo] = res; print(algo, json.dumps(res)); venv.close()
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
