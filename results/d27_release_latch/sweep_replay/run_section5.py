"""D27 Section 5: re-run the adversarial sweep's cases against the D27 code.

Uses section3.py exactly as archived at eefd05e (experiments/adv-sweep/section7-evidence.json,
replay_scripts["section3.py"]; only its two path lines differ). This driver is glue in the same
style as the sweep's own followup3.py: it calls section3's functions (baseline, drops, pushes,
topple, lower, finish) for the cases and adds the D27 info keys to every recorded row.
Outputs: out/s3_*.json (per case, with per-step rows) and summary.json (one row per case).
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import section3 as s

D27_KEYS = ("ever_held", "release_height_mm", "release_xy_error_mm", "release_ok", "is_success",
            "terminated_reason", "piece_on_square", "piece_resting", "gripper_released",
            "piece_settled", "piece_still", "piece_upright")
orig_step = s.Record.step
def d27_step(self, action):
    out = orig_step(self, action)
    self.rows[-1]["d27"] = {k: out[4].get(k) for k in D27_KEYS}
    return out
s.Record.step = d27_step

for pair in (("e4", "e5"), ("b7", "g2")):
    s.drops(pair)                                  # 5, 10, 20, 30, 80 mm (archived function)
for pair in (("e4", "e5"), ("b7", "g2")):
    s.pushes(pair, False)                          # closed push, then finish() opens: push-then-open
s.topple(("b7", "g2"))
for pair in (("c1", "f6"), ("f1", "c6")):          # clean references (as followup3.py)
    r = s.baseline(pair)
    (s.OUT / f"fu_reference_{pair[0]}_{pair[1]}.json").write_text(json.dumps(r))
for mm in (10, 30, 80):                            # c1->f6 drops (as followup3.py)
    pair = ("c1", "f6")
    pre = s.baseline(pair, "traverse"); gate = s.lower(mm / 1000)
    b = s.active.unwrapped; release = b._elapsed_steps
    before = (float(b._achieved_goal()[2]) - b._board_surface_z) * 1000
    s.contact_samples = []; s.sampling = True; s.finish(); s.sampling = False
    s.result(f"fu_drop_{mm}mm", pair, dict(height_gate_reached=gate, release_height_mm=before,
                                          release_step=release, prefix_failure=pre["failure"]))
print("DONE")
