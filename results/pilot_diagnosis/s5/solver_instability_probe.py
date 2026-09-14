"""D14 guard trigger capture for one final model. Read-only: no threshold or guard change.

Replays eval episodes with the same seeding as scripts/eval_final.py. Before every
macro-step the full MjData state is saved; when info['solver_instability'] fires,
that state is restored into a scratch MjData and the same 20 substeps are replayed
one at a time, logging per substep the piece's linear/angular speed, kinetic energy
and every contact touching the piece (geoms, bodies, penetration, normal force).
The replay is checked to reproduce the env's own post-step piece velocity exactly."""
import importlib.util, json, sys
from pathlib import Path
REPO = Path.home()/"projects/robochess_gymnasium"; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO/"scripts/train.py"); train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
run, n_eps, eval_seed, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
cfg = json.loads((run/"manifest.json").read_text())["config_resolved"]; train.apply_thread_pinning(1)
import gymnasium as gym, mujoco, numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.envs import robochess_env as rce
from robochess.tasks.t1_pick_and_place import _pinch_site_id

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
name = lambda m, t, i: mujoco.mj_id2name(m, t, i) or f"#{i}"

class Capture(gym.Wrapper):
    def __init__(s, env):
        super().__init__(env); s.b = env.unwrapped; s.m = s.b.model; s.pin = _pinch_site_id(s.m)
        s.events = []; s.ep = 0; s.scratch = mujoco.MjData(s.m)
    def reset(s, **kw):
        s.ep += 0; return s.env.reset(**kw)
    def piece_stats(s, d):
        v = d.qvel[s.b._piece_vadr:s.b._piece_vadr+6]
        mass = float(s.m.body_mass[s.b._piece_bid]); I = s.m.body_inertia[s.b._piece_bid]
        return dict(lin_speed=float(np.linalg.norm(v[:3])), ang_speed=float(np.linalg.norm(v[3:])))
    def contacts(s, d):
        out = []
        for k in range(d.ncon):
            c = d.contact[k]; b1, b2 = int(s.m.geom_bodyid[c.geom1]), int(s.m.geom_bodyid[c.geom2])
            if s.b._piece_bid not in (b1, b2): continue
            f = np.zeros(6); mujoco.mj_contactForce(s.m, d, k, f)
            og, ob = (c.geom2, b2) if b1 == s.b._piece_bid else (c.geom1, b1)
            out.append(dict(piece_geom=name(s.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1 if b1 == s.b._piece_bid else c.geom2),
                            other_geom=name(s.m, mujoco.mjtObj.mjOBJ_GEOM, og), other_body=name(s.m, mujoco.mjtObj.mjOBJ_BODY, ob),
                            penetration_mm=float(-c.dist*1000), normal_force_N=float(f[0])))
        return out
    def step(s, action):
        d = s.b.data
        pre = np.empty(mujoco.mj_stateSize(s.m, SPEC)); mujoco.mj_getState(s.m, d, pre, SPEC)
        pre_ctrl = None
        pre_stats = s.piece_stats(d); pre_ke = s.b._piece_kinetic_energy(); pre_step = s.b._elapsed_steps
        obs, r, te, tr, info = s.env.step(action)
        if info.get("solver_instability"):
            sd = s.scratch; mujoco.mj_setState(s.m, sd, pre, SPEC); sd.ctrl[:] = d.ctrl; mujoco.mj_forward(s.m, sd)
            subs = []
            for k in range(s.b.frame_skip):
                mujoco.mj_step(s.m, sd)
                st = s.piece_stats(sd)
                # kinetic energy via the env's own formula, evaluated on the scratch data
                real = s.b.data; s.b.data = sd; ke = s.b._piece_kinetic_energy(); s.b.data = real
                subs.append(dict(substep=k+1, **st, ke_J=ke, contacts=s.contacts(sd),
                                 pinch_to_piece_m=float(np.linalg.norm(sd.site_xpos[s.pin]-sd.xpos[s.b._piece_bid]))))
            reproduced = bool(np.allclose(sd.qvel, d.qvel, rtol=0, atol=1e-9))
            first = next((x for x in subs if x["lin_speed"] > rce.PIECE_LIN_SPEED_SANITY or x["ang_speed"] > rce.PIECE_ANG_SPEED_SANITY or x["ke_J"] > rce.PIECE_KINETIC_ENERGY_SANITY), None)
            post = s.piece_stats(d)
            s.events.append(dict(
                eval_episode_index=len(getattr(s, "_done_eps", [])), macro_step=int(info["elapsed_steps"]),
                start=info.get("start_square"), target=info.get("target_square"),
                thresholds=dict(lin=rce.PIECE_LIN_SPEED_SANITY, ang=rce.PIECE_ANG_SPEED_SANITY, ke=rce.PIECE_KINETIC_ENERGY_SANITY),
                pre_step=dict(**pre_stats, ke_J=pre_ke),
                post_step=dict(**post, ke_J=float(info["piece_kinetic_energy_j"]),
                               piece_height_above_board_m=float(info["piece_height_above_board"]),
                               gripper_qpos=float(d.qpos[s.b._grip_qadr]),
                               pinch_to_piece_m=float(np.linalg.norm(d.site_xpos[s.pin]-s.b._achieved_goal())),
                               bad_contact_flag=bool(info["bad_contact"]), piece_off_board_flag=bool(info["piece_off_board"]),
                               contacts=s.contacts(d)),
                which_exceeded=[k for k, hit in (("lin_speed", post["lin_speed"] > rce.PIECE_LIN_SPEED_SANITY),
                                                 ("ang_speed", post["ang_speed"] > rce.PIECE_ANG_SPEED_SANITY),
                                                 ("kinetic_energy", info["piece_kinetic_energy_j"] > rce.PIECE_KINETIC_ENERGY_SANITY)) if hit],
                substep_replay_reproduces_env=reproduced,
                first_substep_over_threshold=first["substep"] if first else None,
                substeps=subs))
        if te or tr:
            s._done_eps = getattr(s, "_done_eps", []) + [1]
        return obs, r, te, tr, info

cap = Capture(train.build_env(cfg))
venv = DummyVecEnv([lambda: Monitor(cap)])
if cfg["algorithm"] == "ppo":
    venv = VecNormalize.load(str(run/"final_vecnormalize.pkl"), venv); venv.training = False; venv.norm_reward = False
model = (PPO if cfg["algorithm"] == "ppo" else SAC).load(str(run/"final_model.zip"), env=venv, device="cpu")
venv.seed(eval_seed); obs = venv.reset(); eps = 0
while eps < n_eps:
    a, _ = model.predict(obs, deterministic=True); obs, r, done, info = venv.step(a); eps += int(done[0])
out.write_text(json.dumps(dict(run=run.name, eval_seed=eval_seed, episodes=n_eps, events=cap.events), indent=1))
print(f"{run.name} seed {eval_seed}: {len(cap.events)} solver_instability events over {n_eps} episodes -> {out}")
