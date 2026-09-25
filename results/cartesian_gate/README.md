# Cartesian expert-replay gate (roadmap 5b.3 step 4 / 5b.8 step E)

Question: can the scripted expert complete pick-and-place through the 4-D Cartesian
action interface (dx, dy, dz, gripper) under the repaired referee? Ten pairs
(e4:e5, a1:h8, h8:a1, a8:h1, h1:h8, h8:h1, c1:f6, f1:c6, b7:g2, d8:e3); d1/e1 are
excluded (D26). Harness: `scripts/gate_expert_cartesian.py` (harness-only; no
environment, referee or expert change). **Only the v3 harness is committed**; v1 and
v2 were never committed, so their JSONs cannot be regenerated from this repo.

## Result: v3 FAILS the gate. Clean place 4/10 (needs 8/10). F2 success 10/10.

| File | Harness | Verdict |
|---|---|---|
| `v1_failed_harness.json` | v1 | 1/10 success. Failures were harness mis-specification (one 60-step timeout on every phase, `held` required at close, 2 mm tolerance at hover waypoints). |
| `v2_failed_place.json` | v2 | 5/10 success. Place target PLACE_Z (0.7935, inherited from the joint expert) sits 5.75 mm below grasp height; head grasps pressed the piece into the board, joint targets wound 50-71 mm of downward lead; opening the pads released it as a plunge (329-505 N): `bad_contact` on c1/f1, release timeouts on a1/h1, e4 lost its grasp in place. The 1.5 mm braking floor kept commanding downward while blocked. |
| `v3_clean_place_gate.json` (+ `v3_stdout.log`) | v3 | 4/10 clean place, 10/10 F2 success. |

## What changed
- **v1 -> v2:** tolerance by phase (2 mm/2 mm/1 deg descend+place, 10 mm/10 mm/2 deg
  transit); close gate = pinched + steady aperture (no `held`), `held` required within
  10 steps in lift; timeouts `ceil(dist/5 mm)+40` for transit, 60 elsewhere; braking
  `max(1.5 mm, min(20 mm, 0.4 d))`; 300-step budget with a 400-step F8 rerun.
- **v2 -> v3:** (a) place target from the live in-hand offset
  (`pinch z - (piece z - rest z) + 1.5 mm`); (b) zero command once inside the position
  tolerance and throughout release; (c) descend braking factor 0.2; (d) release-start
  reporting columns; verdict = clean place (no referee termination AND piece base
  <= 5 mm above resting height at the first release step); F2 success reported separately.

## Disclosures
- **Pre-declared check failed and was withdrawn.** v3 (c) required descend overshoot
  under 5 mm at factor 0.2 (h8, a8) or stop. It failed: h8 19.8 -> 11.2 mm, a8 19.8 ->
  10.9 mm (0.4 -> 0.2). The user withdrew it as a stop condition before any v3 result
  (it was a proxy for grasp height) and replaced it with per-episode columns that do not
  change the verdict. The failure is recorded in `meta.preregistered_check_failed`.
- **Scoring bug fixed between two runs of the same v3 harness.** The first run counted
  `terminated_reason == "success"` as a referee termination, giving clean place 0/10.
  The classifier was corrected to exclude `success` and `time_limit` and the gate
  re-run; trajectories were bit-identical (verified row by row), only the verdict
  changed. The committed JSON is the second run.
- **Interpretation, not stated in the prompt:** the close phase (no tolerance class of
  its own) uses the descend tolerance (2 mm) for the zero-command rule.
- "Piece base" is the piece body-origin height above `_piece_rest_z`, not the lowest
  point of a tilted piece.
- The descend->close gate needs 2 consecutive in-tolerance steps; close->lift needs
  pinched + 2 consecutive steady-aperture steps.
