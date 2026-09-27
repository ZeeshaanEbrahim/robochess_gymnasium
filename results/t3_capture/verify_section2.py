"""Read-only section 2 verification; diagnostic state changes are in memory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import yaml
from scripts.build_t3_scene import SOURCE, GENERATED, HEADER, PAD_BLOCK, CAPTURED_QPOS, captured_body
from scripts.scripted_expert import ScriptedExpert, GRASP_Z, PLACE_Z, GRIPPER_GRIP_CTRL
from robochess.tasks.t3_capture import make_t3_env
from robochess.envs.robochess_env import BOARD_CONTACT_FORCE_CEILING_N, PLACEMENT_REST_TOLERANCE
from robochess import experiment

source = SOURCE.read_bytes()
text = GENERATED.read_bytes().decode()
for block in (HEADER, PAD_BLOCK, captured_body(source.decode()), CAPTURED_QPOS):
    assert text.count(block) == 1
    text = text.replace(block, '', 1)
assert text.encode() == source
print('scene minus additions byte-identical:', True)
models = [mujoco.MjModel.from_xml_path(str(p)) for p in (SOURCE, GENERATED)]
for p, m in zip((SOURCE, GENERATED), models):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'piece_a')
    joint = m.body_jntadr[bid]
    print(p.name, 'piece_a qpos/qvel addresses:', int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint]))
m = models[1]
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'piece_b')
a = int(m.jnt_qposadr[m.body_jntadr[bid]])
print('T3 MuJoCo loads:', True, 'nq:', m.nq, 'keyframe qpos length:', len(m.key_qpos[0]))
print('piece_b jnt_qposadr:', a, 'own keyframe values:', m.key_qpos[0, a:a+7].tolist())
assert len(m.key_qpos[0]) == m.nq
assert np.array_equal(m.key_qpos[0, a:a+7], np.fromstring(CAPTURED_QPOS, sep=' '))
refs = json.loads(Path('results/t3_capture/section2_identity_references.json').read_text())
for path, expected in refs['file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
print('identity source hashes unchanged:', len(refs['file_sha256']))

old = types.ModuleType('robochess._section2_baseline_experiment')
old.__file__ = experiment.__file__
sys.modules[old.__name__] = old
exec(subprocess.check_output(['git', 'show', 'd28-final:robochess/experiment.py']).decode(), old.__dict__)
paths = sorted(Path('configs').glob('*.yaml'))
def resolve(module, raw, path):
    try:
        return ('ok', module.resolve_config(raw, path))
    except module.ConfigError as exc:
        return ('error', str(exc))
for path in paths:
    raw = yaml.safe_load(path.read_text())
    assert resolve(old, raw, path) == resolve(experiment, raw, path), path
print('existing configs compared:', len(paths), 'mismatches:', 0)
for opts in ({'reward_type':'sparse'}, {'her':False}, {'pickup_obs':False}, {'place_reward':False}):
    try:
        make_t3_env(**opts)
    except ValueError as exc:
        print('rejection:', opts, str(exc))
    else:
        raise AssertionError(opts)

env = make_t3_env(start_square='e4', target_square='e5', action_mode='absolute')
b = env.unwrapped
env.reset(seed=0)
expert = ScriptedExpert(env)
b.data.ctrl[-1] = GRIPPER_GRIP_CTRL
# Reuse the expert's existing close budget, keeping its substep cadence.
for _ in range(45 * b.frame_skip):
    mujoco.mj_step(b.model, b.data)
closed = b.data.qpos.copy()
pad = b._geom_id('capture_pad')
print('capture pad body:', mujoco.mj_id2name(b.model, mujoco.mjtObj.mjOBJ_BODY, int(b.model.geom_bodyid[pad])),
      'in board geoms:', pad in b._board_geoms, 'ceiling N:', BOARD_CONTACT_FORCE_CEILING_N)
found = False
for offset in range(10):
    z = PLACE_Z - offset * PLACEMENT_REST_TOLERANCE
    b.data.qpos[:] = closed
    b.data.qvel[:] = 0
    q = expert.solve_ik(np.array([*env.zone_position[:2], z]))
    b.data.qpos[b._arm_qadr] = q
    b.data.ctrl[:6] = q
    mujoco.mj_forward(b.model, b.data)
    forces = []
    for k in range(b.data.ncon):
        c = b.data.contact[k]
        if ((int(c.geom1) in b._robot_geoms and int(c.geom2) == pad)
                or (int(c.geom2) in b._robot_geoms and int(c.geom1) == pad)):
            force = np.zeros(6)
            mujoco.mj_contactForce(b.model, b.data, k, force)
            forces.append(float(abs(force[0])))
    with_pad = b._bad_contact()
    counted = b._last_board_contact_force_n
    original = b._board_geoms.copy()
    try:
        b._board_geoms.discard(pad)
        without_pad = b._bad_contact()
    finally:
        b._board_geoms = original
    print('pad force probe:', {'pinch_z':z, 'pad_contact_count':len(forces), 'pad_normal_sum_N':sum(forces),
                              'base_counted_N':counted, 'bad_with_pad':with_pad,'bad_without_pad':without_pad})
    if sum(forces) > BOARD_CONTACT_FORCE_CEILING_N and with_pad and not without_pad:
        found = True
        break
assert found, 'No isolated above-ceiling pad contact measured'
for height in (0.0, PLACEMENT_REST_TOLERANCE):
    b.data.qpos[:] = closed
    b.data.qvel[:] = 0
    b.data.qpos[env._captured_qadr:env._captured_qadr+7] = [*env.zone_position[:2],env._zone_top,1,0,0,0]
    # Establish a real resting contact under gravity with the arm at home.
    mujoco.mj_forward(b.model,b.data)
    for _ in range(b.frame_skip):
        mujoco.mj_step(b.model,b.data)
    b.data.qpos[env._captured_qadr+2] += height
    b.data.qpos[b._arm_qadr] = expert.solve_ik(np.array([*env.zone_position[:2], GRASP_Z]))
    b.data.qvel[:] = 0
    mujoco.mj_forward(b.model,b.data)
    left,right,board = env._captured_contacts()
    measured_height = float(b.data.xpos[env._captured_bid,2] - env._zone_top)
    print('held probe:', {'lift_from_rest_mm':height*1000,'measured_height_mm':measured_height*1000,
                         'left':left,'right':right,'board':board,'held':left and right and not board})
    assert left and right
    assert board == (height == 0)
env.close()

env = make_t3_env(start_square='e4',target_square='e5',action_mode='cartesian')
obs,info = env.reset(seed=0)
print('smoke reset:', obs['observation'].shape, 'stage:', info['stage'])
ended = []
for i in range(50):
    obs,reward,term,trunc,info = env.step(np.zeros(env.action_space.shape,dtype=np.float32))
    if term or trunc:
        ended.append((i+1,info['terminated_reason']))
print('smoke steps:', i+1, 'observation shape:', obs['observation'].shape,
      'stage:',info['stage'],'observation stage:',float(obs['observation'][-1]),'ended:',ended)
assert obs['observation'].shape == (32,) and info['stage'] == 0 and not ended
env.close()

from robochess.envs.robochess_env import RoboChessEnv, PIECE_LIN_SPEED_SANITY
with make_t3_env(action_mode='cartesian') as env:
    obs, info = env.reset(seed=0)
    b = env.unwrapped
    print('sampling pool:', len(env.reachable_squares), 'excluded:', env.excluded_squares,
          'distinct reset pair:', env.current_start_square != env.current_target_square)
    hooks = ('_is_success', '_piece_off_board', '_solver_instability')
    print('base helper overrides:', [name for name in hooks if name in b.__dict__])
    previous = {key: value.copy() for key, value in obs.items()}
    b.data.qvel[env._captured_vadr] = 2 * PIECE_LIN_SPEED_SANITY
    print('plain captured D14 check:', RoboChessEnv._solver_instability(env._captured_view))
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    same = all(np.array_equal(obs[key], previous[key]) for key in obs)
    print('D14 step:', dict(terminated=terminated, truncated=truncated,
          reason=info['terminated_reason'], success=info['is_success'],
          captured_guard=info['captured_solver_instability'], last_good_returned=same, stage=info['stage']))
    assert truncated and not terminated and not info['is_success'] and same
    assert not any(name in b.__dict__ for name in hooks)
