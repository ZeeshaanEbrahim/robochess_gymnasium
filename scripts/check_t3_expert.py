#!/usr/bin/env python3
"""T3 section 4: unchanged joint expert, read-only diagnosis of the specified pairs.

Interaction stages follow the two controller sequences (remove / move). The
post-latch length is independently total minus the wrapper's stage-1 latch.
Distances include collision geoms only (contype or conaffinity nonzero) on
the gripper and other piece. They are signed MuJoCo surface distances at macro-step
boundaries, matching the existing referee's measurement cadence.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gymnasium as gym
import mujoco
import numpy as np
from robochess.tasks.t3_capture import make_t3_env
from robochess.envs.robochess_env import RoboChessEnv
from scripts.scripted_expert import ScriptedExpert,IKFailure,COMPENSATED_WAYPOINTS

PAIRS=[('c4','f6'),('b3','e7'),('e4','e5'),('d4','c4'),('e4','f5'),('c3','b2'),
       ('g2','h2'),('a2','h2'),('f4','h3'),('b5','h1'),('g3','g2'),('h3','c6'),('h2','a7'),
       ('b6','a8'),('c2','a1'),('g6','h8'),('c1','c4'),('f1','f3'),('h1','h8'),('h8','h1')]

class PieceView:
    """Same controller-local captured-piece settle view used in the positive test."""
    def __init__(self,env):
        self.base=env.unwrapped
        self._piece_bid=env._captured_bid
        self._piece_qadr=env._captured_qadr
        self._piece_vadr=env._captured_vadr
    def __getattr__(self,name):return getattr(self.base,name)


def geom_name(b,g):
    return (mujoco.mj_id2name(b.model,mujoco.mjtObj.mjOBJ_GEOM,g)
            or f'geom_{g}@'+mujoco.mj_id2name(b.model,mujoco.mjtObj.mjOBJ_BODY,int(b.model.geom_bodyid[g])))


def contacts(b):
    return [dict(pair=[geom_name(b,int(c.geom1)),geom_name(b,int(c.geom2))],
                 bodies=[mujoco.mj_id2name(b.model,mujoco.mjtObj.mjOBJ_BODY,int(b.model.geom_bodyid[g]))
                         for g in (c.geom1,c.geom2)],distance_mm=float(c.dist)*1000)
            for c in b.data.contact]


def referee_pairs(b):
    out=[]
    for c in b.data.contact:
        g1,g2=int(c.geom1),int(c.geom2)
        furniture=((g1 in b._robot_geoms and g2 in b._forbidden_geoms)
                   or (g2 in b._robot_geoms and g1 in b._forbidden_geoms))
        self_contact=(g1 in b._robot_geoms and g2 in b._robot_geoms
                      and b.model.geom_bodyid[g1]!=b.model.geom_bodyid[g2]
                      and not b._is_gripper_internal(g1,g2))
        board=((g1 in b._robot_geoms and g2 in b._board_geoms)
               or (g2 in b._robot_geoms and g1 in b._board_geoms))
        if furniture or self_contact or (board and b._last_board_contact_force_n is not None):
            out.append([geom_name(b,g1),geom_name(b,g2)])
    return out


class Observer(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.task=env
        self.phase=None
        self.controller_stage=None
        self.lengths=Counter()
        self.metrics={}
        self.stage_1_step=None
        self.order_event=None
        self.removal_lost_event=None
        self.last_info=None
        self.exempt_steps=[]
        b=self.unwrapped
        root=b._body_id('ur_gripper_base_mount')
        bodies=set()
        for body in range(b.model.nbody):
            parent=body
            while parent:
                if parent==root:
                    bodies.add(body);break
                parent=int(b.model.body_parentid[parent])
        self.gripper_geoms=[g for g in range(b.model.ngeom) if int(b.model.geom_bodyid[g]) in bodies
                            and (b.model.geom_contype[g] or b.model.geom_conaffinity[g])]
        assert self.gripper_geoms

    def begin_stage(self,stage):
        self.controller_stage=stage
        bid=self.unwrapped._piece_bid if stage=='remove' else self.task._captured_bid
        self.other_bid=bid
        self.other_geoms=[g for g in range(self.unwrapped.model.ngeom)
                          if int(self.unwrapped.model.geom_bodyid[g])==bid
                          and (self.unwrapped.model.geom_contype[g]
                               or self.unwrapped.model.geom_conaffinity[g])]
        self.origin=self.unwrapped.data.xpos[bid].copy()
        self.metrics[stage]=dict(min_distance_mm=None,min_pair=None,min_step=None,
                                 peak_displacement_mm=0.,peak_tilt_deg=0.,other_piece='piece_a' if stage=='remove' else 'piece_b')
        self.measure()

    def measure(self):
        b=self.unwrapped;m=b.model;d=b.data
        metric=self.metrics[self.controller_stage]
        metric['peak_displacement_mm']=max(metric['peak_displacement_mm'],float(np.linalg.norm(d.xpos[self.other_bid]-self.origin))*1000)
        tilt=(b._piece_tilt_deg() if self.other_bid==b._piece_bid
              else self.task._for_captured(RoboChessEnv._piece_tilt_deg))
        metric['peak_tilt_deg']=max(metric['peak_tilt_deg'],float(tilt))
        for g in self.gripper_geoms:
            for p in self.other_geoms:
                bound=float(np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[p])+m.geom_rbound[g]+m.geom_rbound[p])
                distance=float(mujoco.mj_geomDistance(m,d,g,p,bound,None))*1000
                if metric['min_distance_mm'] is None or distance<metric['min_distance_mm']:
                    metric.update(min_distance_mm=distance,min_pair=[geom_name(b,g),geom_name(b,p)],
                                  min_step=b._elapsed_steps,min_contacts=contacts(b))

    def step(self,action):
        out=self.env.step(action)
        b=self.unwrapped;info=out[4]
        self.last_info=deepcopy(info)
        if info.get('captured_offboard_exempt'):
            self.exempt_steps.append(dict(step=b._elapsed_steps,
                xy_m=b.data.xpos[self.task._captured_bid,:2].tolist()))
        self.lengths[self.phase]+=1
        self.measure()
        if info['stage']==1 and self.stage_1_step is None:self.stage_1_step=b._elapsed_steps
        if info['order_violation'] and self.order_event is None:
            self.order_event=dict(step=b._elapsed_steps,phase=self.phase,reason=info['order_violation_reason'],contacts=contacts(b))
        if self.controller_stage=='move' and not info['captured_removal'] and self.removal_lost_event is None:
            self.removal_lost_event=dict(step=b._elapsed_steps,phase=self.phase,terms=deepcopy(info['captured_terms']),contacts=contacts(b))
        return out


def run_pair(start,target):
    env=make_t3_env(start_square=start,target_square=target,action_mode='absolute',max_episode_steps=1000)
    obs,info=env.reset(seed=0)
    observer=Observer(env)
    expert=ScriptedExpert(observer)
    stopped=False;exception=None
    try:
        for stage,piece_xy,goal_xy in (
            ('remove',env._board_world_xy(target),env.zone_position[:2]),
            ('move',env._board_world_xy(start),env._board_world_xy(target)),
        ):
            observer.begin_stage(stage)
            expert.base=PieceView(env) if stage=='remove' else env.unwrapped
            previous_target=previous_q=None
            for label,pinch,grip,budget in expert.waypoints(piece_xy,goal_xy):
                observer.phase=f'{stage}:{label}'
                reuse=previous_target is not None and np.allclose(pinch,previous_target)
                try:
                    q=previous_q if reuse else expert.solve_ik(pinch)
                    wp=expert._run_waypoint(pinch,grip,budget,label,None,q,
                                           compensate=label in COMPENSATED_WAYPOINTS and not reuse)
                except IKFailure as exc:
                    exception=str(exc);stopped=True;break
                previous_target,previous_q=pinch.copy(),wp['q_arm']
                info=wp['info']
                if wp['terminated'] or wp['truncated']:
                    stopped=True;break
            if stopped:break
        b=env.unwrapped
        xy=info['captured_release_position']
        captured=dict(height_mm=info['captured_release_height_mm'],xy_m=xy,
                      released=info['captured_terms']['released'],terms=info['captured_terms'])
        if xy is not None:
            xy=np.asarray(xy)
            captured['pad_edge_distances_mm']=dict(left=float(xy[0]-env._zone_low[0])*1000,
                                                  right=float(env._zone_high[0]-xy[0])*1000,
                                                  near=float(xy[1]-env._zone_low[1])*1000,
                                                  far=float(env._zone_high[1]-xy[1])*1000)
        else:captured['pad_edge_distances_mm']=None
        total=b._elapsed_steps
        result=dict(start=start,target=target,success=bool(info['is_success']),max_episode_steps=b.max_episode_steps,
                    stage_1_step=observer.stage_1_step,
                    stage_2_length=total-observer.stage_1_step if observer.stage_1_step is not None else None,
                    total_length=total,phase_lengths=dict(observer.lengths),captured_release=captured,
                    capturing_release=dict(height_mm=info['release_height_mm'],xy_error_mm=info['release_xy_error_mm']),
                    interactions=observer.metrics,gripper_geom_count=len(observer.gripper_geoms),
                    order_event=observer.order_event,removal_lost_event=observer.removal_lost_event,
                    terminated_reason=info['terminated_reason'],off_board_piece=info['off_board_piece'],
                    order_violation=info['order_violation'],captured_removal=info['captured_removal'])
        result['exempt_steps']=observer.exempt_steps
        if not result['success']:
            result['failure']=dict(phase=observer.phase,terminated_reason=info['terminated_reason'],
                                   expert_status='ik_failure' if exception else ('terminated' if info['terminated_reason'] else 'sequence_finished_without_success'),
                                   detail=exception,referee_contact_pairs=referee_pairs(b),active_contacts=contacts(b))
        return result
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/t3_capture/section4_expert.json')
    args=parser.parse_args()
    results=[]
    for start,target in PAIRS:
        result=run_pair(start,target)
        results.append(result)
        print(start+'->'+target,json.dumps({k:result[k] for k in ('success','stage_1_step','stage_2_length','total_length','terminated_reason')}),flush=True)
        if not result['success']:print('FAILURE',json.dumps(result['failure']),flush=True)
        args.out.write_text(json.dumps(dict(results=results),indent=2))
    lengths=[r['total_length'] for r in results if r['success']]
    summary=dict(successes=len(lengths),episodes=len(results),
                 successful_lengths=dict(min=min(lengths),median=statistics.median(lengths),max=max(lengths)) if lengths else None,
                 fit_budgets={str(budget):sum(length<=budget for length in lengths) for budget in (300,400,500,600)})
    args.out.write_text(json.dumps(dict(summary=summary,results=results),indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
