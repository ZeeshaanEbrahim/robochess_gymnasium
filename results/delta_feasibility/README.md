# Is T1 solvable in the delta action space within the episode budget? (18 Sep 2026)

Diagnosis only, run at SHA 97ba93e. No env, config or default was changed. The question is
whether a near-optimal action sequence EXISTS inside the delta action space at the production
200-step budget -- not whether a policy can learn one.

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

## 4. Failure classification: (c), refined (`measure_close_timing.py`, `measure_close_contacts.py`)

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
