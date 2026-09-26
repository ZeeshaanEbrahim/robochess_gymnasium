import sys,json,time
from pathlib import Path
import numpy as np
import gymnasium as gym
import mujoco
REPO=Path(__file__).resolve().parents[3]  # PATH FIX (only change): was /root/projects/robochess_gymnasium
sys.path[:0]=[str(REPO/'scripts'),str(REPO)]
import gate_expert_cartesian as h
from robochess.tasks import PickupReward, PickupObservation
OUT=Path(__file__).resolve().parent/'out'  # PATH FIX (only change): was /tmp/adv_sweep
original_builder=h._build_env
active=None
class Record(gym.Wrapper):
 def __init__(self, shaped):
  self.shaped_value=0.0
  orig=shaped.step
  def shaped_step(action):
   out=orig(action); self.shaped_value=float(out[1]); return out
  shaped.step=shaped_step
  super().__init__(PickupReward(PickupObservation(shaped)))
  self.rows=[]; self.done=False
 def reset(self,**kw):
  self.rows=[]; self.done=False; return self.env.reset(**kw)
 def step(self,action):
  obs,r,t,u,info=self.env.step(action); b=self.unwrapped
  self.done=t or u
  self.rows.append(dict(step=b._elapsed_steps,action=action.tolist(),shaped=self.shaped_value,pickup=float(r),piece=b._achieved_goal().tolist(),pinch=b.data.site_xpos[b._pinch_sid].tolist(),tilt_deg=b._piece_tilt_deg(),speed=float(np.linalg.norm(b.data.qvel[b._piece_vadr:b._piece_vadr+3])),info={k:info.get(k) for k in ['is_success','terminated_reason','piece_on_square','piece_resting','gripper_released','piece_settled','piece_still','piece_upright','grasp_success','pickup_success']}))
  return obs,r,t,u,info
 def close(self): pass

def build(mode,budget):
 global active
 active=Record(original_builder(mode,budget)); return active
h._build_env=build

def baseline(pair,stop=None):
 expert_base=h.RoboChessEnv()
 try: result=h.run_pair(*pair,h.se.ScriptedExpert(expert_base),stop_after=stop)
 finally: expert_base.close()
 result['reward_shaped']=sum(r['shaped'] for r in active.rows)
 result['reward_pickup']=sum(r['pickup'] for r in active.rows)
 result['measurement_rows']=active.rows
 return result

if __name__=='__main__':
 for pair in [('e4','e5'),('b7','g2')]:
  for stop in [None,'traverse']:
   r=baseline(pair,stop)
   label='clean' if stop is None else 'carry'
   (OUT/f's3_{label}_{pair[0]}_{pair[1]}.json').write_text(json.dumps(r))
   print(json.dumps({k:r.get(k) for k in ['start','target','stopped_after','success','clean_place','failure','failure_phase','steps','reward_shaped','reward_pickup','release_start']}),flush=True)

# Substep contact sampling: same integrator calls, split only for measurement.
real_mj_step=mujoco.mj_step
sampling=False
contact_samples=[]
def measured_step(model,data,nstep=1):
 if not sampling: return real_mj_step(model,data,nstep=nstep)
 for sub in range(nstep):
  real_mj_step(model,data,nstep=1)
  mujoco.mj_forward(model,data)
  if h._task_of(active)._piece_contacts()[2]:
   b=active.unwrapped
   contact_samples.append(dict(step=b._elapsed_steps+1,substep=sub,speed=float(np.linalg.norm(data.qvel[b._piece_vadr:b._piece_vadr+3]))))
mujoco.mj_step=measured_step

def command(target,grip,phase,limit=h.FIXED_TIMEOUT,dynamic=None):
 b=active.unwrapped
 streak=0
 for _ in range(limit):
  if active.done: return False
  p=b.data.site_xpos[b._pinch_sid].copy()
  t=dynamic() if dynamic else np.array(target)
  delta=t-p
  inside=np.linalg.norm(delta[:2])<=h.FINE_TOL[0] and abs(delta[2])<=h.FINE_TOL[1]
  mag=h.braked_step_m(float(np.linalg.norm(delta)))
  disp=np.zeros(3) if inside else np.clip(delta/max(np.linalg.norm(delta),1e-12)*mag/b.cartesian_step,-1,1)
  action=np.r_[disp,grip].astype(np.float32)
  active.step(action); active.rows[-1]['phase']=phase
  streak=streak+1 if inside else 0
  if streak>=h.ARRIVE_CONSECUTIVE: return True
 return False

def lower(height):
 b=active.unwrapped
 def target():
  pinch=b.data.site_xpos[b._pinch_sid].copy()
  return np.r_[b._desired_goal[:2],pinch[2]-(float(b._achieved_goal()[2])-b._board_surface_z)+height]
 return command(None,1,'height',dynamic=target)

def finish():
 b=active.unwrapped
 while not active.done:
  active.step(np.array([0,0,0,-1],dtype=np.float32)); active.rows[-1]['phase']='settle'

def result(label,pair,extra):
 b=active.unwrapped; rows=active.rows
 r=dict(scenario=label,start=pair[0],target=pair[1],steps=len(rows),success=rows[-1]['info']['is_success'],reason=rows[-1]['info']['terminated_reason'],final_tilt_deg=b._piece_tilt_deg(),final_upright=rows[-1]['info']['piece_upright'],final_target_distance_mm=float(np.linalg.norm(b._achieved_goal()[:2]-b._desired_goal[:2]))*1000,reward_shaped=sum(x['shaped'] for x in rows),reward_pickup=sum(x['pickup'] for x in rows),**extra)
 (OUT/f's3_{label}_{pair[0]}_{pair[1]}.json').write_text(json.dumps(dict(result=r,rows=rows,contact_samples=contact_samples)))
 print(json.dumps(r),flush=True)
 return r

def drops(pair):
 global sampling,contact_samples
 for mm in [5,10,20,30,80]:
  pre=baseline(pair,'traverse')
  reached=lower(mm/1000)
  b=active.unwrapped
  release_step=b._elapsed_steps
  actual_height=(float(b._achieved_goal()[2])-b._board_surface_z)*1000
  contact_samples=[]; sampling=True
  finish()
  sampling=False
  first_contact=min((s['step'] for s in contact_samples),default=None)
  settled=next((x['step'] for x in active.rows if x['step']>release_step and x['info']['piece_settled'] and x['info']['piece_still'] and x['info']['piece_upright']),None)
  result(f'drop_{mm}mm',pair,dict(height_gate_reached=reached,release_height_mm=actual_height,release_step=release_step,first_board_contact_step=first_contact,peak_piece_speed_at_board_contact=max((s['speed'] for s in contact_samples),default=None),steps_to_settle=None if settled is None else settled-release_step,prefix_failure=pre['failure']))

def pushes(pair,opened):
 global contact_samples
 active=build('cartesian',300); active.reset(seed=0,options={'start_square':pair[0],'target_square':pair[1]})
 b=active.unwrapped; grip=-1 if opened else 1
 # Close/open at home to the existing steady-aperture criterion.
 prev=float(b.data.qpos[b._grip_qadr]); stable=0
 for _ in range(h.FIXED_TIMEOUT):
  active.step(np.r_[np.zeros(3),grip].astype(np.float32)); active.rows[-1]['phase']='gripper_prepare'
  aperture=float(b.data.qpos[b._grip_qadr]); stable=stable+1 if abs(aperture-prev)<h.STEADY_APERTURE_EPS else 0; prev=aperture
  if stable>=h.STEADY_CONSECUTIVE: break
 piece=b._achieved_goal()[:2].copy(); goal=b._desired_goal[:2].copy(); u=(goal-piece)/np.linalg.norm(goal-piece)
 pad=b._geom_id('ur_gripper_right_pad1'); stem=b._geom_id('piece_a_stem'); basegeom=b._geom_id('piece_a_base')
 margin=float(b.model.geom_size[basegeom][0]+b.model.geom_rbound[pad])
 def pad_target(xy,z):
  return lambda: np.r_[xy-(b.data.geom_xpos[pad][:2]-b.data.site_xpos[b._pinch_sid][:2]),z]
 command(None,grip,'approach_push',dynamic=pad_target(piece-u*margin,h.se.CLEAR_Z))
 command(None,grip,'lower_push',dynamic=pad_target(piece-u*margin,h.se.GRASP_Z))
 command(None,grip,'push',limit=300-b._elapsed_steps,dynamic=pad_target(goal-u*float(b.model.geom_size[stem][0]),h.se.GRASP_Z))
 pushed_end=b._achieved_goal().copy()
 # Withdraw vertically with the selected open/closed state, then observe release.
 command(None,grip,'withdraw',dynamic=lambda: np.r_[b.data.site_xpos[b._pinch_sid][:2],h.se.CLEAR_Z])
 finish(); contact_samples=[]
 result('push_open' if opened else 'push_closed',pair,dict(push_end_target_distance_mm=float(np.linalg.norm(pushed_end[:2]-goal))*1000,grasp_success_ever=any(x['info']['grasp_success'] for x in active.rows)))

def topple(pair):
 global contact_samples
 pre=baseline(pair,'traverse'); lowered=lower(0.005)
 b=active.unwrapped
 start_piece=b._achieved_goal().copy()
 # Saturated lateral command while the held piece is low, stopping inside the requested 3 cm region.
 pinch=b.data.site_xpos[b._pinch_sid].copy(); target=pinch.copy(); target[0]+=0.03
 command(target,1,'lateral_bump')
 finish(); contact_samples=[]
 result('topple',pair,dict(height_gate_reached=lowered,prefix_failure=pre['failure'],peak_piece_tilt_deg=max(x['tilt_deg'] for x in active.rows),bump_start_height_mm=(float(start_piece[2])-b._board_surface_z)*1000))
