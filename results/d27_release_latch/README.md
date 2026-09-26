# D27 evidence: placement requires carry and low release (H = 15 mm)

Base commit: `ff7ea92` (master). Code under test: branch `referee-release-latch`. "Old" below always means a
`git worktree` of `ff7ea92`; "new" means the D27 branch. The sweep's simulation baseline `a37c8ab` has the same
`robochess/` source as `ff7ea92` (`git diff --stat a37c8ab ff7ea92 -- robochess` is empty).

## Files and how each was produced

### `section3_equivalence/` (old-vs-new per-step equivalence, `pickup_mode` on)
| file | what |
|---|---|
| `flags_probe.py` | contact flags at reset and after the first hold step, in absolute / delta / Cartesian mode, printed for whichever tree `PYTHONPATH` selects. Old and new print identical flags: `(F,F,F)` at reset, `(F,F,T)` after the first hold step. |
| `equiv_dump.py` | per-step dump of contact flags, D21 reward, `pinched`, `held`, `first_grasp_bonus` and every `pickup_terms` value (floats as `repr`) over four replays: joint e4->e5, joint c1->f6, Cartesian c1->f6 place, Cartesian e4->e5 push. |
| `dump_old.json`, `dump_new.json` | its output on `ff7ea92` and on the D27 branch. Compared row by row: 645 common steps, 0 differing rows, 0 flag mismatches. The push replay has 51 extra steps on the new side because the old code terminated it as a "success" at step 143. On the new side the cached flags also equal a fresh scan on all 696 steps. |

### `section4/` (regressions)
| file | what |
|---|---|
| `run_val.py` | runs the repo's own `fullboard_validation.py` / `fullboard_validation_t2.py` `run()` unchanged (315 episodes: 63 starts x targets e5,a1,h8,a8,h1) and captures each episode's terminal `release_height_mm`, `ever_held`, `release_ok`. `python run_val.py joint OUT.json`, `python run_val.py t2 OUT.json <obstacles>`, run from a tree's root with `PYTHONPATH` set to it. |
| `val_{joint,t2o0,t2o2}_{old,new}.json` | its outputs. |
| `run_gate.py` | the Cartesian v3 gate (`gate_expert_cartesian.run_pair`, the same 10 pairs) on either tree; records the harness clean-place verdict, the env's `is_success`, steps and release height. |
| `gate_{old,new}.json` | its outputs. |
| `compare_section4.py` | prints the tables quoted in DECISIONS.md D27 from those JSON files. |
| `throughput.txt` | `profile_env.py --single --workers 1 --cameras 0 --steps 4000 --warmup 200`, old and new alternated, three repeats each. |

### `sweep_replay/` (Section 5: the adversarial sweep against the D27 code)
| file | what |
|---|---|
| `section3.py` | **extracted verbatim** from `eefd05e:experiments/adv-sweep/section7-evidence.json`, key `replay_scripts["section3.py"]`. Only two lines differ (paths): `sys.path` now uses the repo root computed from `__file__`, and `OUT` is `sweep_replay/out`. |
| `run_section5.py` | glue in the style of the sweep's own `followup3.py`: calls the archived `drops`, `pushes`, `topple`, `baseline`, `lower`, `finish` for the 13 drops, the two clean references and the archived first-version push and topple, and adds the D27 info keys to each row. |
| `recovered/final_section3_probes.py`, `recovered/e4_topple_final.py`, `recovered/followup3.py` | **not in the archive.** Recovered from `/tmp/adv_sweep` on 26 Sep 2026; provenance established by reproduction on `ff7ea92` (below). Paths-only change: a `sys.path` line at the top of each. `followup3.py` is included as the producer of the sweep's `fu_push_then_open` cases; it is not executed itself. |
| `run_recovered.py` | runs the two recovered probe scripts, then builds the push-then-open cases the way `followup3.py` lines 27-47 did (replay the recorded closed-push actions, then open). |
| `compare_gate.py`, `recovered_gate_ff7ea92.json` | the reproduction gate (below). |
| `build_summary.py`, `summary.json` | one row per case: D27 result beside the sweep's recorded values. |
| `sweep_evidence.json`, `sweep_all13_release_events.json` | the sweep's own records, `git show eefd05e:experiments/adv-sweep/evidence.json` and the `all13_release_events.json` entry of `section7-evidence.json`. |
| `out.tar.gz` | the per-case output JSON with per-step rows (extract to `out/`; the directory itself is git-ignored). |

## Reproduction gate for the recovered scripts
The recovered scripts were run on the `ff7ea92` worktree first. All ten outputs equal the sweep's records in
`evidence.json` in steps, success, final tilt, final distance and D20/D21 (`recovered_gate_ff7ea92.json`). In
particular: b7->g2 push-then-open F2 success at step 515, 28.576 mm off target, D20/D21 -183.017/364.971; b7->g2 topple
final tilt 103.965 deg, success False. Only then were they run on the D27 code.

## Archived first versions (labelled, not the sweep's final scenarios)
The archived `section3.py` has earlier versions of two probes: `pushes()` pushes the b7->g2 piece nowhere useful (501 mm
from target) and `topple()` does not topple the piece (peak tilt 4.6 deg). Their D27 runs are in `summary.json` under
"archived first version, not the sweep's final scenario".

## Lesson
The sweep's archive held 6 of its scripts; the final push and topple scripts stayed in `/tmp`. Commit evidence scripts
with the evidence. (`/tmp/adv_sweep` also held the recorded action sequences the push-then-open cases replay; they are
regenerated here by the recovered scripts rather than copied.)
