"""Approved held-carry reruns; all clearances use collision geoms only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from section6_diagnosis import removal
from scripts.check_t3_expert import run_pair, make_t3_env, np
from robochess.envs.robochess_env import SQUARE_PITCH

OUT=Path(__file__).resolve().parent
with make_t3_env(start_square='a8',target_square='h8') as env:
    env.reset(seed=0)
    b=env.unwrapped
    centre=b.data.xpos[b._board_bid,:2].copy()
    half=b.model.geom_size[b._geom_id('board_slab'),:2].copy()

def excursion(row):
    positions=[np.asarray(s['xy_m']) for s in row['exempt_steps']]
    row['max_exempt_excursion_past_board_edge_mm']=max(
        (float(np.max(np.abs(xy-centre)-half))*1000 for xy in positions),default=0.)
    row['max_exempt_excursion_past_offboard_envelope_mm']=max(
        (float(np.max(np.abs(xy-centre)-half-0.5*SQUARE_PITCH))*1000 for xy in positions),default=0.)
    return row

removals=[excursion(removal(square)) for square in ('h7','g8','h8')]
captures=[excursion(run_pair(start,target)) for start,target in
          (('g6','h8'),('h1','h8'),('h8','h1'))]
a8=removal('a8')
result=dict(removals=removals,captures=captures,a8_collision=a8)
(OUT/'section7_diagnosis.json').write_text(json.dumps(result,indent=2))
for r in removals:print('REMOVAL',json.dumps(r))
for r in captures:print('CAPTURE',json.dumps(r))
print('A8_COLLISION',json.dumps(a8))
