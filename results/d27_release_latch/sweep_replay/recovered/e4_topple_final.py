# Recovered from /tmp/adv_sweep on 26 Sep 2026; not in the eefd05e archive; provenance established by
# reproduction on ff7ea92 (see ../README.md). Paths-only change: the sys.path line below, so that
# `import section3` finds the archived section3.py one directory up. The rest is byte-identical to the /tmp copy.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import section3 as s
pair=('e4','e5');s.baseline(pair,'traverse');b=s.active.unwrapped;radius=float(b.model.geom_size[b._geom_id('piece_a_base')][0]);goal=b._desired_goal[:2].copy()
def target():
 pinch=b.data.site_xpos[b._pinch_sid];piece=b._achieved_goal()
 return np.r_[goal-np.array([5*radius,0])-(piece[:2]-pinch[:2]),pinch[2]-(piece[2]-b._board_surface_z)+.020]
s.command(None,1,'pre_bump',dynamic=target)
pad=b._geom_id('ur_gripper_left_pad1')
for _ in range(s.h.FIXED_TIMEOUT):
 if s.active.done:break
 offset=b.data.geom_xpos[pad][:2]-b.data.site_xpos[b._pinch_sid][:2]
 tgt=np.r_[goal-np.array([3*radius,0])-offset,s.h.se.GRASP_Z]
 delta=tgt-b.data.site_xpos[b._pinch_sid]
 s.active.step(np.r_[np.clip(delta/b.cartesian_step,-1,1),-1].astype(np.float32));s.active.rows[-1]['phase']='bump'
 if b._piece_tilt_deg()>90:break
s.command(None,-1,'withdraw',dynamic=lambda:np.r_[b.data.site_xpos[b._pinch_sid][:2],s.h.se.CLEAR_Z]);s.finish();s.contact_samples=[]
s.result('topple_e4_near_target',pair,dict(peak_piece_tilt_deg=max(x['tilt_deg'] for x in s.active.rows)))
