source ~/venvs/robochess/bin/activate && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json,numpy as np,mujoco
from robochess.envs import RoboChessEnv
from scripts.scripted_expert import ScriptedExpert,IKFailure,GRASP_Z,PLACE_Z,CLEAR_Z
from scripts import gate_expert_cartesian as G
candidates=[("right_middle",.351,.52),("left_middle",-.351,.52),("right_near",.351,.325),("right_far",.351,.715),("front_right",.273,.169),("front_left",-.273,.169),("front_center",0,.169),("wide_right",.39,.52),("rear_center",0,.871)]
def selfpairs(b):
 out=[]
 for c in b.data.contact:
  a,z=int(c.geom1),int(c.geom2); m=b.model
  if a in b._robot_geoms and z in b._robot_geoms and m.geom_bodyid[a]!=m.geom_bodyid[z] and not b._is_gripper_internal(a,z):
   out.append([m.body(int(m.geom_bodyid[a])).name,m.body(int(m.geom_bodyid[z])).name,round(float(c.dist)*1000,4)])
 return out
print(json.dumps(dict(kind="method",cart_tolerances=G.TOL,cart_arrive_streak=G.ARRIVE_CONSECUTIVE,budget=G.EPISODE_BUDGET)))
for mode in ["absolute","cartesian"]:
 b=RoboChessEnv(action_mode=mode,max_episode_steps=G.EPISODE_BUDGET);e=ScriptedExpert(b)
 for name,x,y in candidates:
  b.reset(seed=0); row={"mode":mode,"name":name,"phases":[]}
  for label,z in [("approach",CLEAR_Z),("descend",GRASP_Z),("place",PLACE_Z)]:
   target=np.array([x,y,z]); start=int(b._elapsed_steps); seen=[]
   if mode=="absolute":
    try:
     q=e.solve_ik(target)
     result=e._run_waypoint(target,0,25,label,None,q,label=="descend")
     info=result["info"]; term=result["terminated"]; trunc=result["truncated"]
     err=e.pinch_pos()-target
     seen=selfpairs(b)
     rec={"phase":label,"steps":int(b._elapsed_steps)-start,"xy_error_mm":round(float(np.linalg.norm(err[:2]))*1000,3),"z_error_mm":round(float(err[2])*1000,3),"residual_iters":result["iters"],"terminated":term,"reason":info["terminated_reason"],"self_contacts":seen,"bad_contact":info["bad_contact"]}
    except IKFailure as exc:
     rec={"phase":label,"ik_failure":str(exc)}; term=True;trunc=False
   else:
    timeout=G.phase_timeout(label,float(np.linalg.norm(target-b.data.site_xpos[b._pinch_sid])))
    tol=G.TOL["coarse" if label=="approach" else "fine"];streak=0;arrived=False;term=trunc=False
    for _ in range(timeout):
     delta=target-b.data.site_xpos[b._pinch_sid];dist=float(np.linalg.norm(delta))
     inside=np.linalg.norm(delta[:2])<=tol[0] and abs(delta[2])<=tol[1]
     frac=G.BRAKE_FRACTION_DESCEND if label=="descend" else G.BRAKE_FRACTION
     action=np.zeros(4,dtype=np.float32);action[3]=-1
     if not inside and dist>0: action[:3]=np.clip(delta/dist*G.braked_step_m(dist,frac)/b.cartesian_step,-1,1)
     _,_,term,trunc,info=b.step(action)
     seen.extend(selfpairs(b))
     err=b.data.site_xpos[b._pinch_sid]-target
     inside=np.linalg.norm(err[:2])<=tol[0] and abs(err[2])<=tol[1]
     streak=streak+1 if inside else 0
     if term or trunc or streak>=G.ARRIVE_CONSECUTIVE:
      arrived=streak>=G.ARRIVE_CONSECUTIVE;break
    rec={"phase":label,"steps":int(b._elapsed_steps)-start,"xy_error_mm":round(float(np.linalg.norm(err[:2]))*1000,3),"z_error_mm":round(float(err[2])*1000,3),"arrived":arrived,"terminated":term,"reason":info["terminated_reason"],"self_contacts":seen,"bad_contact":info["bad_contact"],"ik_rejects":int(b._ik_rejects)}
   row["phases"].append(rec)
   if term or trunc: break
  print(json.dumps(row),flush=True)
 b.close()
PY
