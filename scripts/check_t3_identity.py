#!/usr/bin/env python3
"""Sequential full-board identity run on an explicitly selected checkout.

Same targets, seed, budget and expert as fullboard_validation{,_t2}.py.
Additionally hashes every step's simulation/observation state and captures
terminal info and contacts, so the diff covers more than rounded summaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--task',choices=('t1','t2_0','t2_2'),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
root=args.repo.resolve()
sys.path.insert(0,str(root/'scripts'))
sys.path.insert(0,str(root))
import gymnasium as gym
import mujoco
import numpy as np
import robochess
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask,T2ObstacleTask,make_t2_env
from robochess.tasks.squares import ALL_SQUARES
from scripted_expert import ScriptedExpert
assert Path(robochess.__file__).resolve().is_relative_to(root)

class Recorder(gym.Wrapper):
    def reset(self,**kwargs):
        self.trace=hashlib.sha256()
        self.last_info=None
        return self.env.reset(**kwargs)
    def step(self,action):
        out=self.env.step(action)
        b=self.unwrapped
        for value in (b.data.qpos,b.data.qvel,b.data.ctrl,*out[0].values()):
            self.trace.update(np.asarray(value).tobytes())
        self.trace.update(json.dumps([out[1],out[2],out[3],out[4]['terminated_reason'],out[4]['is_success']]).encode())
        self.last_info=out[4]
        return out

base=RoboChessEnv(max_episode_steps=350) if args.task=='t1' else make_t2_env(max_episode_steps=350)
task=T1PickAndPlaceTask(base) if args.task=='t1' else T2ObstacleTask(base,n_obstacles=int(args.task[-1]))
recorder=Recorder(task)
expert=ScriptedExpert(recorder)
targets=['e5','a1','h8','a8','h1']
results=[]
started=time.monotonic()
try:
    for target in targets:
        for start in ALL_SQUARES:
            if start==target:continue
            task.start_square,task.target_square=start,target
            result=expert.run_episode(seed=0)
            info=recorder.last_info
            result.update(start=start,target=target,trace_sha256=recorder.trace.hexdigest())
            result['terminal_info']={k:info[k] for k in ('terminated_reason','is_success','ever_held','release_ok',
                                                       'release_height_mm','release_xy_error_mm')}
            if args.task!='t1':
                result['obstacles']={k:info[k] for k in ('obstacles_ever_disturbed','max_obstacle_peak_displacement_mm',
                                                      'obstacles_undisturbed','unintended_displacement_count')}
                result['obstacle_squares']=list(task.current_obstacle_squares)
            if not result['success']:
                result['contacts']=[]
                for contact in base.data.contact:
                    pair=[]
                    for geom in (int(contact.geom1),int(contact.geom2)):
                        body=int(base.model.geom_bodyid[geom])
                        pair.append(dict(geom=geom,name=mujoco.mj_id2name(base.model,mujoco.mjtObj.mjOBJ_GEOM,geom),
                                         body=mujoco.mj_id2name(base.model,mujoco.mjtObj.mjOBJ_BODY,body)))
                    result['contacts'].append(pair)
            results.append(result)
            if len(results)%20==0:
                print(args.task,'episodes',len(results),'successes',sum(r['success'] for r in results),
                      'elapsed_s',round(time.monotonic()-started,2),flush=True)
finally:
    base.close()
args.out.write_text(json.dumps(dict(repo=str(root),task=args.task,results=results),indent=2))
failures=[r for r in results if not r['success']]
print('RESULT',args.task,sum(r['success'] for r in results),'/',len(results),'elapsed_s',round(time.monotonic()-started,2))
for r in failures:
    print('FAILURE',r['start'],r['target'],r['outcome'],r['failed_phase'],r['steps'])
expected={(s,t) for s in ('d1','e1') for t in targets}|{('h1','h8'),('h8','h1')}
assert {(r['start'],r['target']) for r in failures}==expected
assert sum(r['success'] for r in results)==303 and len(results)==315
for r in failures:
    assert r['outcome']==('bad_contact' if r['start'] in ('d1','e1') else 'piece_off_board')
print('Expected failure set and reasons match:',True)
