# Recovered from /tmp/adv_sweep on 26 Sep 2026; not in the eefd05e archive; provenance established by
# reproduction on ff7ea92 (see ../README.md). Paths-only change: the sys.path line below, so that
# `import section3` finds the archived section3.py one directory up. The rest is byte-identical to the /tmp copy.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import section3 as s

def push(pair,opened):
 env=s.build('cartesian',1000);env.reset(seed=0,options={'start_square':pair[0],'target_square':pair[1]})
 b=env.unwrapped;grip=-1 if opened else 1
 prev=float(b.data.qpos[b._grip_qadr]);stable=0
 for _ in range(s.h.FIXED_TIMEOUT):
  env.step(np.r_[np.zeros(3),grip].astype(np.float32));env.rows[-1]['phase']='gripper_prepare'
  aperture=float(b.data.qpos[b._grip_qadr]);stable=stable+1 if abs(aperture-prev)<s.h.STEADY_APERTURE_EPS else 0;prev=aperture
  if stable>=s.h.STEADY_CONSECUTIVE:break
 pc=b._achieved_goal()[:2].copy();goal=b._desired_goal[:2].copy();u=(goal-pc)/np.linalg.norm(goal-pc)
 pad=b._geom_id('ur_gripper_right_pad1');head=b._geom_id('piece_a_head');radius=float(b.model.geom_size[head][0]);margin=float(b.model.geom_size[b._geom_id('piece_a_base')][0]+b.model.geom_rbound[pad])
 def target(xy,z):return np.r_[xy-(b.data.geom_xpos[pad][:2]-b.data.site_xpos[b._pinch_sid][:2]),z]
 s.command(None,grip,'approach',dynamic=lambda:target(pc-u*margin,s.h.se.CLEAR_Z))
 s.command(None,grip,'lower',dynamic=lambda:target(pc-u*margin,b._cartesian_low[2]))
 for _ in range(1000-b._elapsed_steps-s.h.FIXED_TIMEOUT):
  if env.done:break
  pc=b._achieved_goal()[:2].copy();rem=goal-pc
  if np.linalg.norm(rem)<b.goal_tolerance:break
  direction=rem/np.linalg.norm(rem);R=b.data.geom_xmat[pad].reshape(3,3)
  support=float(np.sum(np.abs(np.r_[direction,0]@R)*b.model.geom_size[pad]))
  t=target(pc-direction*(radius+support-s.h.BRAKE_MIN_M),b._cartesian_low[2]);delta=t-b.data.site_xpos[b._pinch_sid]
  dist=np.linalg.norm(delta);mag=s.h.braked_step_m(dist,s.h.BRAKE_FRACTION_DESCEND)
  disp=delta/max(dist,1e-12)*mag/b.cartesian_step
  env.step(np.r_[np.clip(disp,-1,1),grip].astype(np.float32));env.rows[-1]['phase']='push'
 pushed=dict(distance_mm=float(np.linalg.norm(b._achieved_goal()[:2]-goal))*1000,aperture=float(b.data.qpos[b._grip_qadr]),success=env.rows[-1]['info']['is_success'])
 s.command(None,grip,'withdraw',dynamic=lambda:np.r_[b.data.site_xpos[b._pinch_sid][:2],s.h.se.CLEAR_Z])
 while not env.done:
  env.step(np.r_[np.zeros(3),grip].astype(np.float32));env.rows[-1]['phase']='hold'
 s.contact_samples=[]
 s.result('push_open_final' if opened else 'push_closed_final',pair,dict(push_end=pushed,grasp_success_ever=any(r['info']['grasp_success'] for r in env.rows),budget=1000))

def topple(pair):
 s.baseline(pair,'traverse');b=s.active.unwrapped
 radius=float(b.model.geom_size[b._geom_id('piece_a_base')][0]);goal=b._desired_goal[:2].copy()
 def lower_target():
  pinch=b.data.site_xpos[b._pinch_sid];piece=b._achieved_goal()
  return np.r_[goal+np.array([radius,0])-(piece[:2]-pinch[:2]),pinch[2]-(piece[2]-b._board_surface_z)+.020]
 s.command(None,1,'pre_bump',dynamic=lower_target)
 pad=b._geom_id('ur_gripper_right_pad1')
 for _ in range(s.h.FIXED_TIMEOUT):
  if s.active.done:break
  offset=b.data.geom_xpos[pad][:2]-b.data.site_xpos[b._pinch_sid][:2]
  target=np.r_[goal-np.array([radius,0])-offset,s.h.se.GRASP_Z]
  delta=target-b.data.site_xpos[b._pinch_sid]
  s.active.step(np.r_[np.clip(delta/b.cartesian_step,-1,1),-1].astype(np.float32));s.active.rows[-1]['phase']='bump'
  if b._piece_tilt_deg()>90:break
 s.command(None,-1,'withdraw',dynamic=lambda:np.r_[b.data.site_xpos[b._pinch_sid][:2],s.h.se.CLEAR_Z])
 s.finish();s.contact_samples=[]
 s.result('topple_final',pair,dict(peak_piece_tilt_deg=max(x['tilt_deg'] for x in s.active.rows)))
for pair in [('e4','e5'),('b7','g2')]:
 push(pair,False);push(pair,True);topple(pair)
