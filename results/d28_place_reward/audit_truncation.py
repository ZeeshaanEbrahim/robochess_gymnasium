"""D28 audit A1: does a 300-step truncation on a place-mode env reach both learners as a
time-limit (bootstrapped), not a terminal?  Run as a script (building SB3 models inside the
pytest process segfaults torch in the full suite -- see DECISIONS D28).

    python audit_truncation.py
"""
import importlib.util, sys, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
warnings.simplefilter("ignore")
import gymnasium as gym, numpy as np, stable_baselines3 as sb3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from robochess.experiment import resolve_config
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
print("stable-baselines3", sb3.__version__)

def cfg(algo, **kw):
    return resolve_config(dict(task="t1", algorithm=algo, reward_type="shaped", observation="state", seed=0,
        total_timesteps=1, action_mode="cartesian", max_episode_steps=300, start_square="e4", target_square="e5",
        pickup_obs=True, place_obs=True, place_mode=True, device="cpu", **kw), config_path="t.yaml")

# ---- (a) DummyVecEnv marks the truncation; no gym TimeLimit in the stack
c = cfg("ppo"); venv = train.build_vec_env(c)
node, stack = venv.venv.envs[0], []
while node is not None:
    stack.append(type(node).__name__); node = getattr(node, "env", None)
print("wrapper stack:", stack)
assert "TimeLimit" not in stack
venv.reset(); zero = np.zeros((1, 4), dtype=np.float32)
for k in range(1, 301):
    obs, r, done, infos = venv.step(zero)
    if done[0]: break
print(f"(a) episode ended at step {k}: done={bool(done[0])} TimeLimit.truncated={infos[0]['TimeLimit.truncated']} "
      f"terminated_reason={infos[0].get('terminated_reason')} has terminal_observation={'terminal_observation' in infos[0]}")
assert k == 300 and infos[0]["TimeLimit.truncated"] is True

# ---- (b) PPO bootstraps the value on it: buffer reward = env reward + gamma * V(terminal obs), only there
model = PPO(policy="MultiInputPolicy", env=venv, seed=0, device="cpu", n_steps=600, batch_size=100, verbose=0)
logged, calls = [], []
orig_sw = venv.step_wait
def sw():
    out = orig_sw(); logged.append(float(out[1][0])); return out
venv.step_wait = sw
orig_pv = model.policy.predict_values
def pv(o):
    v = orig_pv(o); calls.append(float(v[0])); return v
model.policy.predict_values = pv
venv.reset(); model._last_obs = venv.reset(); model._last_episode_starts = np.ones((1,), dtype=bool)
class CB(sb3.common.callbacks.BaseCallback):
    def _on_step(self): return True
import collections
model.ep_info_buffer = collections.deque(maxlen=100); model.ep_success_buffer = collections.deque(maxlen=100)
cb = CB(); cb.init_callback(model)
model.collect_rollouts(venv, cb, model.rollout_buffer, n_rollout_steps=600)
buf = model.rollout_buffer.rewards[:, 0]
diff = buf - np.array(logged[-600:])
nz = np.flatnonzero(np.abs(diff) > 1e-9)
print(f"(b) rollout of 600 steps: buffer-minus-env reward is nonzero at (0-based) steps {nz.tolist()} (1-based: {[int(i)+1 for i in nz]})")
# predict_values is called only for the truncation bootstraps (rollout values come from forward())
# plus once for the value of the last obs: 2 truncations + 1 = 3 calls
print(f"    predict_values calls: {len(calls)} (2 truncation bootstraps + 1 final-obs value)")
assert list(nz) == [299, 599] and len(calls) == 3
assert abs(diff[299] - model.gamma * calls[0]) < 1e-6 and abs(diff[599] - model.gamma * calls[1]) < 1e-6
print(f"    exact: buffer - env reward at step 300 = gamma*V(terminal obs) = {model.gamma}*{calls[0]:.6f} = {model.gamma*calls[0]:.6f}; "
      f"step 600: {model.gamma*calls[1]:.6f}; zero at every other step")
venv.close()

# ---- (c) SAC replay buffer: timeouts stored, and sampled as not-done
c = cfg("sac"); venv = train.build_vec_env(c)
model = SAC(policy="MultiInputPolicy", env=venv, seed=0, device="cpu", learning_starts=100000, verbose=0)
rb = model.replay_buffer
print("(c) replay buffer:", type(rb).__name__, "handle_timeout_termination =", rb.handle_timeout_termination)
model.learn(total_timesteps=300)
print(f"    step 300 stored: dones={bool(rb.dones[299,0])} timeouts={bool(rb.timeouts[299,0])}; other steps timeouts any={bool(rb.timeouts[:299].any())}")
s = rb._get_samples(np.array([299]), env=model._vec_normalize_env)
print(f"    value sampled for the truncation transition: dones={float(s.dones[0])}  (bootstrapped: target uses (1 - dones))")
assert rb.handle_timeout_termination and bool(rb.timeouts[299, 0]) and bool(rb.dones[299, 0]) and float(s.dones[0]) == 0.0
print("A1: PASS (a)(b)(c)")
