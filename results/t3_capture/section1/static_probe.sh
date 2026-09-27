source ~/venvs/robochess/bin/activate && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json, numpy as np, mujoco
from robochess.envs import RoboChessEnv
from robochess.envs import robochess_env as R
from scripts.scripted_expert import ScriptedExpert, IKFailure, GRASP_Z, PLACE_Z, CLEAR_Z
b=RoboChessEnv(action_mode="cartesian"); b.reset(seed=0); e=ScriptedExpert(b)
m,d=b.model,b.data
board=m.body_pos[b._body_id("board")].copy(); half=m.geom_size[b._geom_id("board_slab")][:2]
candidates=[]
for side in [-1,1]:
 for y in [board[1]-2.5*R.SQUARE_PITCH,board[1],board[1]+2.5*R.SQUARE_PITCH,board[1]+3.5*R.SQUARE_PITCH]:
  candidates.append((f"side_{side}_{y:.3f}", side*b._cartesian_high[0],y))
for x in [-3.5*R.SQUARE_PITCH,0,3.5*R.SQUARE_PITCH]:
 candidates.append((f"front_{x:.3f}",x,b._cartesian_low[1]))
 candidates.append((f"back_{x:.3f}",x,b._cartesian_high[1]))
for x in [-1,1]:
 candidates.append((f"wide_side_{x}",x*(half[0]+R.SQUARE_PITCH),board[1]))
candidates.append(("far_back",0,b._cartesian_high[1]+.5*R.SQUARE_PITCH))
def contacts():
 out=[]
 for c in d.contact:
  g1,g2=int(c.geom1),int(c.geom2)
  if g1 in b._robot_geoms and g2 in b._robot_geoms and m.geom_bodyid[g1]!=m.geom_bodyid[g2] and not b._is_gripper_internal(g1,g2):
   out.append([m.body(int(m.geom_bodyid[g1])).name,m.body(int(m.geom_bodyid[g2])).name,round(float(c.dist)*1000,4)])
 return out
def clip(target):
 saved=d.site_xpos[b._pinch_sid].copy()
 d.site_xpos[b._pinch_sid]=target
 used=[]
 original=b._solve_cartesian_ik
 b._solve_cartesian_ik=lambda t: used.append(t.copy()) or None
 try: b._scale_action(np.array([0,0,0,-1]))
 finally: b._solve_cartesian_ik=original; d.site_xpos[b._pinch_sid]=saved
 return used[0]
print(json.dumps(dict(kind="geometry",board_center=board.tolist(),physical_half=half.tolist(),cart_low=b._cartesian_low.tolist(),cart_high=b._cartesian_high.tolist(),goal_tolerance=b.goal_tolerance,piece_radius=float(m.geom_size[b._geom_id("piece_a_base")][0]),heights={"grasp":GRASP_Z,"place":PLACE_Z,"clear":CLEAR_Z},f2={"rest":R.PLACEMENT_REST_TOLERANCE,"linear":R.PIECE_LIN_SETTLE_TOL,"angular":R.PIECE_ANG_SETTLE_TOL,"tilt":R.PIECE_TILT_TOL_DEG,"streak":R.PIECE_SETTLE_CONSECUTIVE_STEPS})))
for name,x,y in candidates:
 row=dict(name=name,xy=[round(x,5),round(y,5)],radius=round(float(np.hypot(x,y)),5),heights={})
 for label,z in [("grasp",GRASP_Z),("place",PLACE_Z)]:
  b.reset(seed=0); target=np.array([x,y,z]); used=clip(target); rec={"clipped_mm":round(float(np.linalg.norm(used-target))*1000,3),"used":np.round(used,5).tolist()}
  try:
   q=e.solve_ik(target); d.qpos[b._arm_qadr]=q; mujoco.mj_forward(m,d)
   rec.update(ik=True,error_mm=round(float(np.linalg.norm(d.site_xpos[b._pinch_sid]-target))*1000,4),self_contacts=contacts(),joint_limits=bool(np.all(q>=b._ctrl_low[:6]) and np.all(q<=b._ctrl_high[:6])))
   rec["production_ik_at_solution"]=b._solve_cartesian_ik(target) is not None
   rec["radial_limits"]=[round(float(np.interp(z,R.CARTESIAN_INNER_Z,R.CARTESIAN_INNER_R)),5),min(R.CARTESIAN_MAX_RADIUS,float(np.interp(z,R.CARTESIAN_REACH_Z,R.CARTESIAN_REACH_R))-R.CARTESIAN_REACH_MARGIN)]
  except IKFailure as exc: rec.update(ik=False,error=str(exc))
  row["heights"][label]=rec
 print(json.dumps(row),flush=True)
b.close()
PY
