# Replaying the expert's commands through the delta action space (18 Sep 2026)

Diagnosis only. No env, config or default was changed. Sections 1-4 were run at SHA 97ba93e;
sections 5-7 (schedule-shifted replay, pre-clamp comparison) at cddd5d0.

WHAT THIS MEASURES: whether the SCRIPTED EXPERT'S OWN RECORDED COMMAND SEQUENCE, replayed
through the delta action space at the production 200-step budget, reaches the goal -- and where
it stops doing so. It does not measure whether the action space admits a solution in general,
and it does not measure whether a policy can learn one. The expert's sequence carries a fixed
SCHEDULE (each waypoint's duration was fixed by settle checks in a faster absolute-mode run);
a closed-loop policy has no such schedule, so a replay failure at a schedule boundary says
something about the recording, not about the action space.

Scripts: `scripts/replay_expert_delta.py` (the reusable converter + replay harness) and the
four measurement scripts here. Run each from the repo root with the venv active.

## 1. Achievable joint speed (`measure_joint_speed.py` -> `joint_speed.json`)

Macro step 40 ms (timestep 0.002 x frame_skip 20); production budget 200 steps = 8.0 s.
Sustained full-magnitude delta from home, one joint at a time; measured, not computed from gains.

| joint | plateau +1 (rad/s) | plateau -1 | limiting | travel per 200 steps (rad) | (deg) |
|---|---|---|---|---|---|
| shoulder_pan | 0.630 | 0.630 | 0.630 | 5.04 | 289 |
| shoulder_lift | 0.716 | 0.690 | 0.690 | 5.52 | 316 |
| elbow | 0.658 | 0.643 | 0.643 | 5.14 | 295 |
| wrist_1 | 0.452 | 0.451 | 0.451 | 3.60 | 207 |
| wrist_2 | 0.455 | 0.455 | 0.455 | 3.64 | 209 |
| wrist_3 | 0.465 | 0.465 | 0.465 | 3.72 | 213 |

The plateau is set by the servo, not the action space: a 0.1 rad cap per 40 ms step is a
commanded 2.5 rad/s, but the joints reach 0.45-0.72 rad/s, so during sustained motion the
target sits permanently at the clamp margin ahead of the joint. Three of twelve sweeps stall
on collision rather than a speed limit (shoulder_lift +1 pads into the board slab at step 28,
shoulder_lift -1 an arm link into the table at 72, elbow +1 a self-collision at 46); their
plateaus are taken over the free-motion phase before contact.

## 2. Travel the task requires (`measure_expert_travel.py` -> `expert_travel.json`)

Scripted expert, 10 starts spanning the envelope, target e5, all 10 succeed. Summed |dq| per
joint from home. Worst cell is wrist_1 on d1 at 0.60x the section-1 ceiling through place;
every other cell is <= 0.57x. NO joint is over budget on any square.

Separately (steps, not travel): the expert needs 113-184 steps to reach place and 127-198 for a
full episode, so on a8/h8 only 2 steps of the 200-step budget are spare. Those steps are spent
settling, not moving, so they do not appear as travel.

## 3. Replay through the RL action space (`scripts/replay_expert_delta.py` -> `replay_200.json`)

The expert's per-step absolute commands are recorded from a normal expert run, then converted
online to delta actions (`to_delta_action`) and fed to `scripts/train.py`'s own `build_env` in
delta mode at max_episode_steps=200 -- the same path SB3 uses. **6/10 succeed** (a8, h8, d8, b4,
g6, c7); a1, h1, d1, f2 fail with no grasp, closest approach 0.039-0.041 m and final distance
0.24-0.44 m. Every episode ended by exhausting its command sequence, none by truncation.

## 4. Failure classification under the expert's own schedule: (c), refined (`measure_close_timing.py`, `measure_close_contacts.py`)

Not (a): 131-138 of 200 steps used. Not (b): tracking is inaccurate exactly where it matters.

The commanded target DOES catch up -- median lag 0.000 rad on nearly every joint (h1 wrist_3,
0.275, excepted), maxima 0.3-1.1 rad right after each waypoint jump -- but it catches up 8-27
steps late, and the expert's command sequence is open-loop in time. When the close command
fires, the failing squares' pinch site is 0.069-0.085 m from the piece against the expert's own
0.036, so the gripper shuts on air: aperture runs to 0.766-0.781 with no pad ever touching the
piece, versus 0.650 with both pads on it for b4. The rest of the carry runs with an empty gripper.

The failing squares are the ones with a SHORT descend phase (24-28 steps vs a8's 45): the arm's
achievable speed cannot cover the descend in that window. Over a1's 32 descend+close steps a
joint wanted a full 0.1 rad step on 28 and the windup clamp cut the step short on all 32 (b4:
19 and 22 of 34) -- both bind, the clamp more tightly, and the clamp binds because the joint
itself cannot keep up.

NOT measured: whether these four squares would succeed with the clamp absent, which would
separate "the clamp margin limits this" from "the servo speed limits this". That needs the
pre-clamp code and was out of scope for this diagnosis.

## 5. Schedule-shifted replay (`--close-gate` -> `replay_200_close_gate.json`)

Harness-only change: hold the final descend command until the pinch reaches the distance the
expert's own pinch had when IT commanded the close, then run the rest of the sequence unchanged.
Wait capped at the episode's remaining steps. Env untouched.

**7/10 succeed (was 6/10), 8/10 grasp.** Of section 3's four failures, d1 and f2 now SUCCEED and
a1 now grasps -- each needed a wait of 10-15 steps against the 62-69 they had spare. The
schedule, not the action space and not the budget, was the operative cause on three of four.

The two that still fail are not action-space limits either:
- a1 grasps, carries and places to 15.6 mm XY, resting and released; it fails only on
  `piece_settled` (piece still at 7.65 rad/s) when the recorded sequence ends at 141 steps with
  59 unused. The recorded release tail is shorter than the settling needs.
- h1: holding the descend command pushes the PIECE 29.9 mm sideways, so the gate distance
  becomes unreachable. At steady state the delta target equals the expert's command on every
  joint, with no lag, no force saturation and the arm at rest -- the arm is where it was told to
  be; the piece is not.
- c7 is a HARNESS artifact and a regression caused by this gate: it plateaus 0.79 mm outside a
  gate it previously passed by closing immediately.

## 6. Does the clamp margin limit joint speed (`measure_joint_speed_preclamp.py`)

Yes, by 2.8-5.0x. Pre-clamp (worktree at b434fac), same procedure:

| joint | clamped (rad/s) | pre-clamp | ratio |
|---|---|---|---|
| shoulder_pan | 0.630 | 2.278 | 3.62x |
| shoulder_lift | 0.690 | 1.922 | 2.79x |
| elbow | 0.643 | 1.950 | 3.03x |
| wrist_1 | 0.451 | 2.268 | 5.03x |
| wrist_2 | 0.455 | 2.253 | 4.95x |
| wrist_3 | 0.465 | 2.258 | 4.86x |

This CORRECTS section 1's claim that the servo rather than the action space sets the plateau:
unclamped, the joints reach 1.9-2.3 rad/s, near the 2.5 rad/s the 0.1 rad cap allows per 40 ms
step. Section 1's stall labels also over-attribute to collision -- at 2.3 rad/s most pre-clamp
sweeps are decelerating into the joint's own range limit (stopping 0.22 rad short of +-6.28),
and only shoulder_lift is genuinely obstructed.

## 7. Schedule-shifted replay on the pre-clamp code (`replay_200_close_gate_preclamp.json`)

**Pre-clamp 6/10 vs clamped 7/10.** Removing the clamp does not fix section 3's failures and
breaks b4 and f2 instead. But read this comparison carefully: every pre-clamp gate miss is by
1.2-2.2 mm and clamped c7's by 0.79 mm -- the gate threshold is knife-edge, not a physical
wall. Only h1's 9.89 mm miss is substantive. Both configurations put the pinch within ~2 mm of
the expert's own close distance on 9 of 10 squares.

## 8. Step budget evidence

Maximum steps consumed by any SUCCESSFUL episode:

| configuration | successes | max steps |
|---|---|---|
| original replay, clamped, expert schedule | 6/10 | 198 |
| close-gate, clamped | 7/10 | 199 |
| close-gate, pre-clamp | 6/10 | 199 |

Scripted expert in its own harness (350-step budget), same 10 squares, all successful:
a1 131, a8 198, h1 133, h8 198, d1 138, d8 194, b4 127, g6 145, c7 181, f2 133 -- max 198.

No budget value is recommended here.

## Conclusion (what was measured, not a general feasibility claim)

Replaying the expert's recorded commands through the delta action space at 200 steps reaches
6/10 squares on the expert's own schedule and 7/10 when the close is gated on arrival instead of
on the timer. The four schedule failures had 62-69 unused steps against a shortfall of 8-27, so
neither the budget nor the action space explains them; three of the four convert on a 10-15 step
wait. The residual failures are a recorded release tail too short for the piece to settle (a1),
the held descend pushing the piece (h1), and a knife-edge gate threshold in this harness (c7).
An earlier version of this file concluded "T1 is not cleanly solvable in the delta action space
at the 200-step budget" -- that claim was not supported by these measurements and has been
withdrawn.
