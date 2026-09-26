"""D27 Section 5, recovered scenarios. The tree under test is the one three directories up from this file.

1. runs recovered/final_section3_probes.py and recovered/e4_topple_final.py exactly as recovered
   (push closed, push open, topple, for e4->e5 and b7->g2; then the e4 near-target topple);
2. builds the sweep's "push-then-open" cases the way recovered/followup3.py did (its lines 27-47,
   reproduced here because executing followup3.py also re-runs everything else): replay the recorded
   closed-push actions up to (not including) the 'hold' phase, then finish() (open the gripper and settle).
   e4->e5 uses the archived first-version push (s3_push_closed_e4_e5), b7->g2 the recovered final one
   (s3_push_closed_final_b7_g2), exactly as followup3.py did.
Adds the D27 info keys (absent on ff7ea92: recorded as None) to each row, like run_section5.py.
Run order matters: this script needs out/s3_push_closed_e4_e5.json from run_section5.py (archived pushes())
when it is run in a fresh out/; run_recovered.py therefore runs the archived push first if it is missing.
"""
import json, runpy, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import section3 as s
KEYS = ("ever_held", "release_height_mm", "release_xy_error_mm", "release_ok", "is_success", "terminated_reason",
        "piece_on_square", "piece_resting", "gripper_released", "piece_settled", "piece_still", "piece_upright")
orig_step = s.Record.step
def d27_step(self, action):
    out = orig_step(self, action)
    self.rows[-1]["d27"] = {k: out[4].get(k) for k in KEYS}
    return out
s.Record.step = d27_step
if not (s.OUT / "s3_push_closed_e4_e5.json").exists():
    s.pushes(("e4", "e5"), False)           # archived first-version push; followup3.py replays its actions
runpy.run_path(str(HERE / "recovered" / "final_section3_probes.py"), run_name="recovered_final")
runpy.run_path(str(HERE / "recovered" / "e4_topple_final.py"), run_name="recovered_e4_topple")
for pair in (("e4", "e5"), ("b7", "g2")):
    if pair[0] == "e4":
        old = json.loads((s.OUT / "s3_push_closed_e4_e5.json").read_text()); actions = old["rows"]
    else:
        old = json.loads((s.OUT / "s3_push_closed_final_b7_g2.json").read_text()); actions = []
        for row in old["rows"]:
            if row.get("phase") == "hold": break
            actions.append(row)
    env = s.build("cartesian", 1000); env.reset(seed=0, options={"start_square": pair[0], "target_square": pair[1]})
    for row in actions:
        if env.done: break
        env.step(np.array(row["action"], dtype=np.float32)); env.rows[-1]["phase"] = row.get("phase")
    if not env.done: s.finish()
    s.contact_samples = []
    s.result("fu_push_then_open", pair, dict(source_actions=len(actions)))
print("DONE")
