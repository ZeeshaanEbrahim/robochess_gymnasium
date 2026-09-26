# Recovered from /tmp/adv_sweep on 26 Sep 2026; not in the eefd05e archive; provenance established by
# reproduction on ff7ea92 (see ../README.md). Paths-only change: the sys.path line below, so that
# `import section3` finds the archived section3.py one directory up. The rest is byte-identical to the /tmp copy.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np
import section3 as s
orig=s.Record.step
def rich_step(self,action):
 out=orig(self,action);b=self.unwrapped;task=s.h._task_of(self)
 left,right,board=task._piece_contacts()
 row=self.rows[-1]
 row.update(aperture=float(b.data.qpos[b._grip_qadr]),height_mm=(float(b._achieved_goal()[2])-b._board_surface_z)*1000,pinched=bool(left and right),held=bool(left and right and not board),pad_contact=bool(left or right),piece_board=bool(board),ang_speed=float(np.linalg.norm(b.data.qvel[b._piece_vadr+3:b._piece_vadr+6])),success_terms=list(b._success_terms()))
 return out
s.Record.step=rich_step

def events(rows,open_step):
 crossing=next((r for r in rows if r['step']>=open_step and r['aperture']<.2),None)
 last=next((r for r in reversed(rows) if r['step']>=open_step and r['pad_contact']),None)
 return dict(open_command_step=open_step,qpos_crossing_step=None if crossing is None else crossing['step'],height_at_qpos_crossing_mm=None if crossing is None else crossing['height_mm'],last_pad_contact_step=None if last is None else last['step'],height_at_last_pad_contact_mm=None if last is None else last['height_mm'])

for pair in [('c1','f6'),('f1','c6')]:
 r=s.baseline(pair)
 (s.OUT/f'fu_reference_{pair[0]}_{pair[1]}.json').write_text(json.dumps(r))
 print('reference',json.dumps({k:r.get(k) for k in ['start','target','success','clean_place','steps','reward_shaped','reward_pickup','release_start']}),flush=True)
for mm in [10,30,80]:
 pair=('c1','f6');pre=s.baseline(pair,'traverse');gate=s.lower(mm/1000)
 b=s.active.unwrapped;release=b._elapsed_steps
 before=(float(b._achieved_goal()[2])-b._board_surface_z)*1000
 s.contact_samples=[];s.sampling=True;s.finish();s.sampling=False
 rows=s.active.rows
 r=s.result(f'fu_drop_{mm}mm',pair,dict(height_gate_reached=gate,release_height_mm=before,release_step=release,**events(rows,release+1)))
for pair in [('e4','e5'),('b7','g2')]:
 if pair[0]=='e4':
  old=json.loads((s.OUT/'s3_push_closed_e4_e5.json').read_text());actions=old['rows']
 else:
  old=json.loads((s.OUT/'s3_push_closed_final_b7_g2.json').read_text());actions=[]
  for row in old['rows']:
   if row.get('phase')=='hold':break
   actions.append(row)
 env=s.build('cartesian',1000);env.reset(seed=0,options={'start_square':pair[0],'target_square':pair[1]})
 for row in actions:
  if env.done:break
  env.step(np.array(row['action'],dtype=np.float32));env.rows[-1]['phase']=row.get('phase')
 if not env.done:s.finish()
 rows=env.rows;opens=[r for r in rows if r['action'][-1]<1]
 opening=next((r['step'] for r in rows if r.get('phase')=='settle' and r['action'][-1]==-1),None)
 push=next((r['step'] for r in rows if r.get('phase') in ('push','base_push')),None)
 success=next((r for r in rows if r['info']['is_success']),None)
 s.contact_samples=[]
 r=s.result('fu_push_then_open',pair,dict(push_start_step=push,opening_step=opening,f2_success_step=None if success is None else success['step'],max_piece_lift_mm=max(x['height_mm'] for x in rows),pinched_ever=any(x['pinched'] for x in rows),held_ever=any(x['held'] for x in rows),grasp_success_ever=any(x['info']['grasp_success'] for x in rows),f2_at_success=success,**events(rows,opening or 1)))
