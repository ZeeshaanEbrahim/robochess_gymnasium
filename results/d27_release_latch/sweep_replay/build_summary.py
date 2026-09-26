"""Build summary.json (one row per Section 5 case) from out/ (run_section5.py + run_recovered.py on the D27 code)
and the sweep's recorded values (sweep_evidence.json, sweep_all13_release_events.json). Prints the table."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
ev = json.load(open(H / "sweep_evidence.json"))
a13 = {(e["case"], e["pair"]): e for e in json.load(open(H / "sweep_all13_release_events.json"))}
OTH = ("piece_on_square", "piece_resting", "gripper_released", "piece_settled", "piece_still", "piece_upright")
def load(f):
    o = json.load(open(H / "out" / f)); return o.get("measurement_rows") or o["rows"]
def row(name, f, label, sweep_key=None, a13key=None):
    rows = load(f); last = rows[-1]["d27"]
    first6 = next((r["step"] for r in rows if all(r["d27"][k] for k in OTH)), None)
    c = dict(case=name, label=label, steps=len(rows), success=last["is_success"], reason=last["terminated_reason"],
             ever_held=last["ever_held"], release_ok=last["release_ok"], release_height_mm=last["release_height_mm"],
             release_xy_error_mm=last["release_xy_error_mm"], failing_terms_other_than_release_ok=[k for k in OTH if not last[k]],
             first_step_all_six_other_terms_hold=first6, final_tilt_deg=json.load(open(H / "out" / f)).get("result", {}).get("final_tilt_deg"),
             D20=sum(x["shaped"] for x in rows), D21=sum(x["pickup"] for x in rows))
    sw = ev.get(sweep_key or f, {}).get("result")
    if sw:
        c.update(sweep_steps=sw["steps"], sweep_success=sw["success"], sweep_D20=sw["reward_shaped"], sweep_D21=sw["reward_pickup"])
    if a13key in a13:
        e = a13[a13key]; c.update(sweep_last_pad_height_mm=e["last_pad_height_mm"], sweep_length=e["length"])
    return c
ARCH, REC = "archived section3.py (eefd05e)", "recovered from /tmp (reproduced on ff7ea92)"
S = []
for pair in ("e4_e5", "b7_g2"):
    for mm in (5, 10, 20, 30, 80):
        S.append(row(f"drop {mm} mm {pair}", f"s3_drop_{mm}mm_{pair}.json", ARCH, a13key=(f"drop_{mm}mm", pair.replace("_", "→"))))
for mm in (10, 30, 80):
    S.append(row(f"drop {mm} mm c1_f6", f"s3_fu_drop_{mm}mm_c1_f6.json", ARCH + ", followup3.py glue",
                 a13key=(f"fu_drop_{mm}mm", "c1→f6")))
for pair in ("c1_f6", "f1_c6"):
    S.append(row(f"clean reference {pair}", f"fu_reference_{pair}.json", ARCH))
S.append(row("push-then-open e4_e5", "s3_fu_push_then_open_e4_e5.json", REC + ": replay of archived first-version closed push (as followup3.py)"))
S.append(row("push-then-open b7_g2", "s3_fu_push_then_open_b7_g2.json", REC + ": final closed push + open"))
S.append(row("topple b7_g2 (final)", "s3_topple_final_b7_g2.json", REC))
S.append(row("topple e4_e5 (final)", "s3_topple_final_e4_e5.json", REC))
S.append(row("topple e4_e5 near target", "s3_topple_e4_near_target_e4_e5.json", REC))
S.append(row("archived first version: closed push b7_g2 (501 mm)", "s3_push_closed_b7_g2.json", "archived first version, not the sweep's final scenario"))
S.append(row("archived first version: topple b7_g2 (does not topple)", "s3_topple_b7_g2.json", "archived first version, not the sweep's final scenario"))
json.dump(S, open(H / "summary.json", "w"), indent=1)
for c in S:
    print(f"{c['case']:58s} steps={c['steps']:4d} success={c['success']!s:5s} ever_held={c['ever_held']!s:5s} release_ok={c['release_ok']!s:5s} "
          f"rel_h={c['release_height_mm'] if c['release_height_mm'] is None else round(c['release_height_mm'], 3)} sweep_last_pad={c.get('sweep_last_pad_height_mm')} "
          f"other_failing={c['failing_terms_other_than_release_ok']}")
