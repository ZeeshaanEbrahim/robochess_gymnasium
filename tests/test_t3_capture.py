"""T3 contracts and real/constructed capture controls (section 3)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from robochess.envs.robochess_env import (
    RoboChessEnv, BOARD_CONTACT_FORCE_CEILING_N, PLACEMENT_RELEASE_MAX_HEIGHT,
    PLACEMENT_REST_TOLERANCE, PIECE_LIN_SPEED_SANITY,
)
from robochess.experiment import ConfigError, resolve_config
from robochess.tasks.t3_capture import make_t3_env, T3CaptureTask, reject_t3_training_options
from scripts.build_t3_scene import SOURCE, GENERATED, CAPTURED_QPOS
from scripts.scripted_expert import ScriptedExpert, GRASP_Z, PLACE_Z, COMPENSATED_WAYPOINTS

ROOT = Path(__file__).resolve().parents[1]
PAIR = ('c4', 'f6')

@pytest.fixture
def task():
    with make_t3_env(start_square=PAIR[0], target_square=PAIR[1], action_mode='absolute') as env:
        env.reset(seed=0)
        yield env


def snapshot(env):
    b = env.unwrapped
    state = np.empty(mujoco.mj_stateSize(b.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(b.model, b.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    fields = ('stage', 'order_violation', '_order_reason', '_captured_ever_held',
              '_captured_release_record', '_captured_streaks', '_captured_terms',
              '_captured_removal', '_off_board_piece', '_captured_offboard_exempt',
              '_capturing_start_xyz', '_last_good_t3_obs')
    base_fields = ('_elapsed_steps', '_ever_held', '_release_record', '_contact_flags',
                   '_piece_ang_settled_streak', '_piece_lin_settled_streak', '_piece_upright_streak',
                   '_last_good_obs', '_arm_target')
    return state, {k: deepcopy(getattr(env,k)) for k in fields}, {k: deepcopy(getattr(b,k)) for k in base_fields}


def restore(env, saved):
    state, own, base = saved
    b = env.unwrapped
    mujoco.mj_setState(b.model,b.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    for k,v in own.items(): setattr(env,k,deepcopy(v))
    for k,v in base.items(): setattr(b,k,deepcopy(v))
    mujoco.mj_forward(b.model,b.data)


class PieceView:
    """Controller-local address view; no environment handle or hook is changed."""
    def __init__(self, env):
        self.base = env.unwrapped
        self._piece_bid = env._captured_bid
        self._piece_qadr = env._captured_qadr
        self._piece_vadr = env._captured_vadr
    def __getattr__(self, name):
        return getattr(self.base, name)


class Recorder(gym.Wrapper):
    def __init__(self, task):
        super().__init__(task)
        self.rows = []
        self.phase = None
    def step(self, action):
        out = self.env.step(action)
        info = out[4]
        self.rows.append(dict(step=self.unwrapped._elapsed_steps, phase=self.phase,
                              stage=info['stage'], success=info['is_success'],
                              reason=info['terminated_reason'],
                              captured_release_height_mm=info['captured_release_height_mm'],
                              capturing_release_height_mm=info['release_height_mm']))
        return out


def real_capture():
    """Physical: unchanged joint waypoints, IK, compensation and release ramp."""
    # Two existing full-board budgets; this is a test cap, not T3's selected budget.
    env = make_t3_env(start_square=PAIR[0], target_square=PAIR[1], action_mode='absolute', max_episode_steps=2*350)
    obs,info = env.reset(seed=0)
    recorder = Recorder(env)
    expert = ScriptedExpert(recorder)
    states = {'reset': snapshot(env)}
    try:
        for stage, piece_xy, goal_xy in (
            ('captured', env._board_world_xy(PAIR[1]), env.zone_position[:2]),
            ('capturing', env._board_world_xy(PAIR[0]), env._board_world_xy(PAIR[1])),
        ):
            expert.base = PieceView(env) if stage == 'captured' else env.unwrapped
            previous_target = previous_q = None
            for label, target, grip, budget in expert.waypoints(piece_xy,goal_xy):
                recorder.phase = f'{stage}:{label}'
                reuse = previous_target is not None and np.allclose(target, previous_target)
                q = previous_q if reuse else expert.solve_ik(target)
                wp = expert._run_waypoint(target, grip, budget, label, None, q,
                                         compensate=label in COMPENSATED_WAYPOINTS and not reuse)
                previous_target, previous_q = target.copy(), wp['q_arm']
                info = wp['info']
                states[f'{stage}:{label}'] = snapshot(env)
                if wp['terminated'] or wp['truncated']:
                    contacts = [(int(c.geom1),int(c.geom2),float(c.dist)) for c in env.unwrapped.data.contact]
                    return dict(success=info['is_success'], rows=recorder.rows, states=states,
                                info=deepcopy(info), stopped_phase=recorder.phase, contacts=contacts)
        return dict(success=info['is_success'], rows=recorder.rows, states=states,
                    info=deepcopy(info), stopped_phase=recorder.phase)
    finally:
        env.close()


@pytest.fixture(scope='module')
def positive():
    result = real_capture()
    printable = {k:v for k,v in result.items() if k not in ('states','info')}
    printable['info'] = {k:result['info'][k] for k in ('stage','is_success','terminated_reason',
                         'captured_terms','captured_release_height_mm','release_height_mm','order_violation')}
    out = ROOT/'results/t3_capture/section3_positive.json'
    out.write_text(json.dumps(printable,indent=2))
    assert result['success'], printable
    return result


def test_keyframe_addresses_and_length():
    original = mujoco.MjModel.from_xml_path(str(SOURCE))
    generated = mujoco.MjModel.from_xml_path(str(GENERATED))
    def addresses(m, name):
        b = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
        j = m.body_jntadr[b]
        return int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j])
    assert addresses(original,'piece_a') == addresses(generated,'piece_a')
    q,_ = addresses(generated,'piece_b')
    declarations = re.findall(r'<key\s+name="home"\s+qpos="([^"]+)"', GENERATED.read_text())
    assert len(declarations) == 1
    declared = np.fromstring(declarations[0], sep=' ')
    assert len(declared) == generated.nq
    np.testing.assert_array_equal(declared, generated.key_qpos[0])
    np.testing.assert_array_equal(declared[q:q+7],np.fromstring(CAPTURED_QPOS,sep=' '))
    assert len(generated.key_qpos[0]) == generated.nq
    np.testing.assert_array_equal(generated.key_qpos[0,q:q+7],np.fromstring(CAPTURED_QPOS,sep=' '))


def test_observation_target_reset_and_pool(task):
    obs,info = task.reset(seed=0)
    b = task.unwrapped
    assert obs['observation'].shape == (32,)
    assert task.observation_space.contains(obs)
    assert info['stage'] == obs['observation'][-1] == 0
    np.testing.assert_allclose(b.data.xpos[task._captured_bid,:2],task._board_world_xy(PAIR[1]))
    assert task.current_target_square != 'e5'
    assert task.excluded_squares == ('d1','e1')
    assert len(task.reachable_squares) == 62
    assert not {'d1','e1'} & set(task.reachable_squares)
    for sq in task.excluded_squares:
        with pytest.raises(ValueError,match='excluded'):
            task.reset(options={'start_square':sq})
        with pytest.raises(ValueError,match='excluded'):
            task.reset(options={'target_square':sq})


def close_at_home(task):
    b = task.unwrapped
    b.data.ctrl[-1] = 255
    for _ in range(45*b.frame_skip): mujoco.mj_step(b.model,b.data)
    return b.data.qpos.copy()


def test_pad_counted_as_board_force_and_held(task):
    """Constructed poses; contacts and their forces are real MuJoCo outputs."""
    b = task.unwrapped
    expert = ScriptedExpert(task)
    closed = close_at_home(task)
    pad = b._geom_id('capture_pad')
    assert int(b.model.geom_bodyid[pad]) == b._board_bid and pad in b._board_geoms
    assert BOARD_CONTACT_FORCE_CEILING_N == 250
    b.data.qpos[b._arm_qadr] = expert.solve_ik(np.array([*task.zone_position[:2],PLACE_Z-3*PLACEMENT_REST_TOLERANCE]))
    b.data.qvel[:] = 0
    b.data.ctrl[:6] = b.data.qpos[b._arm_qadr]
    mujoco.mj_forward(b.model,b.data)
    forces=[]
    for k,c in enumerate(b.data.contact):
        if ((int(c.geom1) in b._robot_geoms and int(c.geom2)==pad) or
            (int(c.geom2) in b._robot_geoms and int(c.geom1)==pad)):
            wrench=np.zeros(6);mujoco.mj_contactForce(b.model,b.data,k,wrench)
            forces.append(abs(wrench[0]))
    assert sum(forces)>BOARD_CONTACT_FORCE_CEILING_N
    assert b._bad_contact()
    assert b._last_board_contact_force_n == pytest.approx(sum(forces))
    original=b._board_geoms.copy()
    try:
        b._board_geoms.discard(pad)
        assert not b._bad_contact()
    finally:
        b._board_geoms=original
    assert pad in b._board_geoms
    for height in (0.,PLACEMENT_REST_TOLERANCE):
        b.data.qpos[:]=closed;b.data.qvel[:]=0
        b.data.qpos[task._captured_qadr:task._captured_qadr+7]=[*task.zone_position[:2],task._zone_top,1,0,0,0]
        mujoco.mj_forward(b.model,b.data)
        for _ in range(b.frame_skip):mujoco.mj_step(b.model,b.data)
        b.data.qpos[task._captured_qadr+2]+=height
        b.data.qpos[b._arm_qadr]=expert.solve_ik(np.array([*task.zone_position[:2],GRASP_Z]))
        b.data.qvel[:]=0;mujoco.mj_forward(b.model,b.data)
        task._read_captured_terms(advance=True)
        terms=task._augment_info(b._build_info(None))['captured_terms']
        assert terms['pinched']
        assert terms['board_contact'] == (height==0)
        assert terms['held'] == (height!=0)


def test_captured_d14_spike_returns_last_good(task):
    """Constructed spike through a real env step; no methods are patched."""
    obs,_=task.reset(seed=0);b=task.unwrapped
    b.data.qvel[task._captured_vadr]=2*PIECE_LIN_SPEED_SANITY
    after,_,term,trunc,info=task.step(ScriptedExpert(task)._to_action(b.data.ctrl[:6],b.data.ctrl[-1]))
    assert trunc and not term
    assert info['terminated_reason']=='solver_instability'
    assert info['captured_solver_instability'] and info['invalid_transition']
    assert not info['is_success'] and info['stage']==0
    for key in obs: np.testing.assert_array_equal(after[key],obs[key])
    assert not any(k in b.__dict__ for k in ('_is_success','_solver_instability','_piece_off_board'))


@pytest.mark.parametrize('key,value,message',[
    ('reward_type','sparse','T3 reward not implemented'),('reward',False,'T3 reward not implemented'),
    ('pickup_mode','pickup','T3 reward not implemented'),('shaped_staged',False,'T3 reward not implemented'),
    ('her',False,'HER not supported'),('her_enabled',True,'HER not supported'),('use_her',True,'HER not supported'),
    ('replay_buffer_class','HerReplayBuffer','HER not supported'),
    ('pickup_obs',False,'pickup_obs or place-reward keys'),('place_reward',False,'pickup_obs or place-reward keys'),
    ('place_mode',False,'pickup_obs or place-reward keys'),
])
def test_all_training_options_rejected(key,value,message,task):
    opts={key:value}
    with pytest.raises(ValueError,match=message): make_t3_env(**opts)
    with pytest.raises(ValueError,match=message): T3CaptureTask(task.unwrapped,**opts)
    with pytest.raises(ValueError,match=message): task.reset(options=opts)
    with pytest.raises(ConfigError,match=message): resolve_config({'task':'t3',**opts})


def test_training_config_and_her_reward_rejected(task):
    with pytest.raises(ConfigError,match='T3 reward not implemented'):resolve_config({'task':'t3'})
    with pytest.raises(ValueError,match='HER not supported'):task.compute_reward(None,None,None)
    with pytest.raises(TypeError,match='Unknown T3'):T3CaptureTask(task.unwrapped,unknown=True)


def test_identity_references_unchanged():
    refs=json.loads((ROOT/'results/t3_capture/section2_identity_references.json').read_text())
    for path,expected in refs['file_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path


def test_physical_capture_succeeds(positive):
    """Physical positive control: chained joint expert on c4 -> f6."""
    rows=positive['rows'];info=positive['info']
    latch=next(row['step'] for row in rows if row['stage']==1)
    success=next(row['step'] for row in rows if row['success'])
    assert latch<success
    assert info['terminated_reason']=='success'
    assert info['captured_terms']['release_ok'] and info['release_ok']
    print('physical positive:',dict(pair=PAIR,stage_latch_step=latch,success_step=success,
          captured_release_height_mm=info['captured_release_height_mm'],capturing_release_height_mm=info['release_height_mm']))


def put_piece(task, captured, xyz, quat=(1,0,0,0)):
    b=task.unwrapped
    q=task._captured_qadr if captured else b._piece_qadr
    v=task._captured_vadr if captured else b._piece_vadr
    b.data.qpos[q:q+7]=[*xyz,*quat]
    b.data.qvel[v:v+6]=0
    mujoco.mj_forward(b.model,b.data)
    b._refresh_contact_flags()


def quiet_step(task):
    b=task.unwrapped
    return task.step(ScriptedExpert(task)._to_action(b.data.ctrl[:6],b.data.ctrl[-1]))


def unremoved(task):
    task.stage=0
    task._captured_removal=False
    task._captured_ever_held=False
    task._captured_release_record=None
    task._captured_streaks=[0,0,0]
    task.order_violation=False
    task._order_reason=None


def test_captured_off_board_outside_pad(task,positive):
    """Constructed: change the reset state's captured destination to outside pad."""
    restore(task,positive['states']['reset'])
    put_piece(task,True,[task._zone_high[0]+2*task._captured_radius,task.zone_position[1],task._zone_top])
    _,_,term,trunc,info=quiet_step(task)
    assert term and not trunc
    assert info['terminated_reason']=='piece_off_board'
    assert info['off_board_piece']=='piece_b' and info['piece_off_board']
    assert not info['is_success']


def test_captured_pushed_into_pad_without_grasp(task,positive):
    """Constructed: positive destination, but omit the grasp/release history."""
    restore(task,positive['states']['captured:release'])
    unremoved(task)
    from robochess.envs.robochess_env import PIECE_SETTLE_CONSECUTIVE_STEPS
    for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS+1):
        _,_,_,_,info=quiet_step(task)
        assert not info['captured_removal'] and not info['is_success']
        assert not info['captured_terms']['ever_held'] and not info['captured_terms']['release_ok']
    assert all(info['captured_terms'][k] for k in ('in_zone','resting','still','settled','upright','released'))


def test_captured_release_too_high(task,positive):
    """Constructed: only the positive release record's height is raised."""
    restore(task,positive['states']['captured:release'])
    task.stage=0
    record=task._captured_release_record
    task._captured_release_record=(*record[:2],2*PLACEMENT_RELEASE_MAX_HEIGHT)
    _,_,_,_,info=quiet_step(task)
    terms=info['captured_terms']
    assert info['captured_release_height_mm']>PLACEMENT_RELEASE_MAX_HEIGHT*1000
    assert terms['ever_held'] and not terms['release_ok']
    assert all(terms[k] for k in ('in_zone','resting','still','settled','upright','released'))
    assert not info['captured_removal'] and not info['is_success']


def test_wrong_order_capturing_gripped_first(task,positive):
    """Constructed: replay the real capturing grasp before removal instead of after."""
    restore(task,positive['states']['capturing:close'])
    unremoved(task)
    put_piece(task,True,[*task._board_world_xy(PAIR[1]),task._piece_rest_z])
    _,_,_,_,info=quiet_step(task)
    assert info['order_violation']
    assert info['order_violation_reason']=='capturing_piece_pinched'
    assert not info['is_success']


def test_capturing_placed_while_captured_on_square(task,positive):
    """Constructed: positive capturing placement with captured removal omitted."""
    restore(task,positive['states']['capturing:place'])
    unremoved(task)
    put_piece(task,True,[*task._board_world_xy(PAIR[1]),task._piece_rest_z])
    # Observe the constructed order event before integrating overlapping pieces.
    task._update_order()
    _,_,_,_,info=quiet_step(task)
    assert info['order_violation']
    assert info['order_violation_reason'] in ('capturing_piece_pinched','capturing_piece_displaced')
    assert not info['captured_removal'] and not info['is_success']


def test_captured_toppled_in_pad(task,positive):
    """Constructed: change only the positive captured placement's orientation."""
    restore(task,positive['states']['captured:release'])
    put_piece(task,True,task.zone_position,(np.sqrt(.5),np.sqrt(.5),0,0))
    _,_,_,_,info=quiet_step(task)
    assert info['captured_terms']['in_zone']
    assert not info['captured_terms']['upright']
    assert info['captured_terms']['tilt_deg']>10
    assert not info['captured_removal'] and not info['is_success']


def test_captured_in_pad_still_held(task,positive):
    """Physical checkpoint: same positive path, stopped before place and release."""
    restore(task,positive['states']['captured:traverse'])
    _,_,_,_,info=quiet_step(task)
    assert info['captured_terms']['in_zone']
    assert info['captured_terms']['held'] and not info['captured_terms']['released']
    assert not info['captured_removal'] and not info['is_success']


def test_capturing_piece_on_pad_has_no_exemption(task,positive):
    """Constructed: pad destination applied to capturing instead of captured."""
    restore(task,positive['states']['reset'])
    put_piece(task,False,task.zone_position)
    _,_,term,trunc,info=quiet_step(task)
    assert term and not trunc
    assert info['piece_off_board'] and info['off_board_piece']=='piece_a'
    assert info['terminated_reason']=='piece_off_board'
    assert not info['is_success']


def test_stage_remains_latched_after_removal_lost(task,positive):
    restore(task,positive['states']['captured:release'])
    assert task.stage==1
    put_piece(task,True,[*task._board_world_xy(PAIR[1]),task._piece_rest_z])
    _,_,_,_,info=quiet_step(task)
    assert info['stage']==1
    assert not info['captured_removal'] and not info['is_success']


def test_base_success_is_not_final_success(task,positive):
    restore(task,positive['states']['capturing:release'])
    task._captured_ever_held=False
    task._captured_release_record=None
    _,_,_,_,info=quiet_step(task)
    assert info['capturing_t1_success']
    assert not info['captured_removal'] and not info['is_success']
    assert info['terminated_reason'] is None


def test_physical_high_drop_over_pad(positive):
    """Physical: release from the actual traverse checkpoint; omit low place."""
    with make_t3_env(start_square=PAIR[0],target_square=PAIR[1],
                     action_mode='absolute',max_episode_steps=2*350) as env:
        env.reset(seed=0)
        restore(env,positive['states']['captured:traverse'])
        expert=ScriptedExpert(env)
        expert.base=PieceView(env)
        b=env.unwrapped
        # Keep the existing traverse joint target and use the unchanged release
        # ramp/budget. No pose, velocity, contact or history is constructed.
        release=next(w for w in expert.waypoints(env._board_world_xy(PAIR[1]),
                                                env.zone_position[:2]) if w[0]=='release')
        wp=expert._run_waypoint(expert.pinch_pos(),release[2],release[3],
                               'release',None,b.data.ctrl[:6].copy(),compensate=False)
        info=wp['info']
        assert info['captured_ever_held']
        assert info['captured_terms']['released']
        assert info['captured_release_height_mm']>PLACEMENT_RELEASE_MAX_HEIGHT*1000
        assert not info['captured_terms']['release_ok']
        assert not info['captured_removal'] and info['stage']==0
        measurement=dict(release_height_mm=info['captured_release_height_mm'],
                         removal=info['captured_removal'],stage=info['stage'],reason=info['terminated_reason'])
        (ROOT/'results/t3_capture/section3b_high_drop.json').write_text(json.dumps(measurement,indent=2))
        print('physical high drop:',measurement)


def test_physical_pinched_while_resting(positive):
    """Physical checkpoint: low place finished, before the release ramp."""
    with make_t3_env(start_square=PAIR[0],target_square=PAIR[1],
                     action_mode='absolute',max_episode_steps=2*350) as env:
        env.reset(seed=0)
        restore(env,positive['states']['captured:place'])
        _,_,_,_,info=quiet_step(env)
        terms=info['captured_terms']
        assert terms['in_zone'] and terms['resting']
        assert terms['pinched'] and terms['board_contact']
        assert not terms['released']
        assert not info['captured_removal'] and info['stage']==0
        measurement=dict(pinched=terms['pinched'],board_contact=terms['board_contact'],
                         height_mm=terms['height_mm'],released=terms['released'],removal=info['captured_removal'],stage=info['stage'])
        (ROOT/'results/t3_capture/section3b_pinched_resting.json').write_text(json.dumps(measurement,indent=2))
        print('physical pinched resting:',measurement)


def test_physical_held_capture_crosses_edge_and_succeeds():
    """Physical: unchanged chained joint expert, formerly failing h8 removal."""
    from scripts.check_t3_expert import run_pair
    result = run_pair('g6', 'h8')
    assert result['exempt_steps']
    assert result['success'] and result['terminated_reason'] == 'success'
    assert result['captured_removal']


def test_captured_released_outside_board_over_table(task):
    """Constructed airborne release outside both allowed footprints."""
    put_piece(task, True, [0.45, 0.52, 0.82])
    _, _, terminated, truncated, info = quiet_step(task)
    assert terminated and not truncated
    assert info['terminated_reason'] == 'piece_off_board'
    assert info['off_board_piece'] == 'piece_b'
    assert not info['captured_offboard_exempt']
    assert info['captured_terms']['released']


def test_captured_pinched_on_table_outside_board(task):
    """Constructed drag pose; bilateral finger and table contacts are real."""
    b = task.unwrapped
    close_at_home(task)
    xy = np.array([0.36, 0.52])
    table = b._geom_id('table')
    top = b.data.geom_xpos[table, 2] + b.model.geom_size[table, 2]
    put_piece(task, True, [*xy, top])
    for _ in range(b.frame_skip):
        mujoco.mj_step(b.model, b.data)
    q = ScriptedExpert(task).solve_ik(np.array([*xy, GRASP_Z + top - task._zone_top]))
    b.data.qpos[b._arm_qadr] = q
    b.data.ctrl[:6] = q
    b.data.qvel[:] = 0
    mujoco.mj_forward(b.model, b.data)
    assert all(task._captured_contacts()[:2])
    assert any(table in (int(c.geom1), int(c.geom2)) and
               task._captured_bid in (int(b.model.geom_bodyid[c.geom1]),
                                     int(b.model.geom_bodyid[c.geom2]))
               for c in b.data.contact)
    assert task._either_piece_off_board()
    assert not task._captured_offboard_exempt
    _, _, terminated, truncated, info = quiet_step(task)
    assert terminated and not truncated
    assert info['terminated_reason'] == 'piece_off_board'
    assert info['off_board_piece'] == 'piece_b'
    assert info['captured_terms']['pinched']
    assert not info['captured_offboard_exempt']
