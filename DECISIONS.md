# Decisions

Design decisions with lasting consequences, and the measurements behind them.
Newest last. Scene files carry a one-line pointer here rather than repeating the
rationale.

---

## D1 — Board scale: 78 mm squares, not the 57 mm tournament standard

**Date:** 2026-08-16
**Status:** accepted
**Evidence:** `assets/scale_test.xml` (a standalone two-row comparison scene),
`mj_geomDistance` and contact-penetration measurements against the
`robotiq_2f85` model.

### Decision

Chess squares are **78 mm** centre-to-centre. The playing surface is therefore
624 x 624 mm.

### Rationale

Measured from the 2F-85 model, not nominal spec:

Max open width is **85.2 mm** pad-face to pad-face, at the driver joints' q=0
limit (`range "0 0.8"` rad, `ctrl=0`). Nominal spec is 85 mm, so the model
matches.

But 85.2 mm is the INNER pad gap, and it is **not** the binding number. The
follower links flare outwards well beyond the pads, and their widest point sits
at roughly the height of a 50 mm piece's top rim. Simulated result, gripper
centred on the middle piece of each row, penetration into the NEIGHBOURING
piece:

| pitch | ctrl=0 | ctrl=110 | ctrl=170 | ctrl=255 | |
| --- | --- | --- | --- | --- | --- |
| 57 mm | 1.45 mm | 2.32 mm | 2.63 mm | 2.63 mm | fouls at EVERY aperture |
| 78 mm | none | none | none | none | clean at every aperture |

At 57 mm there is no aperture that avoids the neighbours: closing further does
not help, because it is the follower links, not the pads, that collide. Both
rows do grasp their target piece successfully from ctrl~170.

### Alternative considered and rejected

The foul is at the neighbour's TOP RIM, so piece height is the other lever.
Holding 57 mm pitch and 30 mm diameter, the neighbour clears once the pieces are
**<= ~42 mm tall** (46 mm still grazes by 0.4 mm).

Rejected: a 42 mm ceiling on piece height breaks piece proportions for the king
and queen, which need to be visibly taller than a pawn. Giving up realistic
piece proportions is a worse trade than giving up tournament-standard pitch,
since the pitch is invisible to the policy while the proportions are exactly
what distinguishes the piece classes.

---

## D2 — Reach envelope: all 64 squares reachable, corners at ~99% of rated reach

**Date:** 2026-08-16
**Status:** accepted, with a flagged risk
**Evidence:** `assets/scene.xml`; damped-least-squares IK sweep over all 64
squares with the approach axis locked to world-down.

### Decision

Scene layout is fixed at:

| | value |
| --- | --- |
| board centre | **x = 0, y = 0.52** |
| board playing surface | z = 0.762 (12 mm slab on a table top at z = 0.750) |
| arm base | **x = 0, y = 0, z = 0.85**, on a pedestal standing on the table |
| grasp height used for the sweep | z = 0.792 (mid-height of a 50 mm piece) |

### Rationale

With the arm base at 0.85 and the board centre at y=0.52, a top-down IK solve
succeeds for **all 64 squares**, with no arm/environment interpenetration in any
of the solutions. Nearest square d1 sits at 0.250 m from the base axis; farthest
corner square h8 at **0.839 m**, against the UR5e's 850 mm rated reach — so the
far corners are near full extension.

Moving the board closer does NOT help: the near squares fail first (they end up
underneath the shoulder), which is what fixes y at 0.52. A sweep over arm base
heights 0.85–1.00 and board distances 0.48–0.56 confirmed 0.85 / 0.52 as the
best placement that still keeps every square inside the rated envelope.

### Flagged risk

The corner squares have almost no margin. Anything that lengthens the required
reach will push them out:

- any future **off-axis / non-top-down approach** (T2 obstacles, T4/T5), which
  trades vertical clearance for horizontal extension;
- real-world slack — mounting tolerance, table position error, a physical
  gripper slightly longer than the model.

Revisit this decision if manipulability problems show up in training — poor
corner-square success rates, or IK failures near the h/a files. Levers available
at that point, roughly in order of cost: a tilted corner approach, a smaller
board, a second arm base position, or a longer-reach arm.

---

## D3 — Goal representation: the tracked piece's 3D position

**Date:** 2026-08-16
**Status:** accepted
**Implemented by:** `robochess/envs/robochess_env.py` (P1 scaffold)

### Decision

The HER goal is the **tracked piece's 3D world position** — a 3-vector — not an
occupancy grid or a board-state encoding. Both `achieved_goal` and
`desired_goal` are `Box(3,)`, and `achieved_goal` is read directly from the
piece body's `xpos`.

This was already the working assumption for the project's HER setup; it is
recorded here because P1 locks it into the observation space rather than
leaving it open.

### Rationale

Hindsight Experience Replay relabels a failed episode by substituting a goal
that *was* achieved. That only works when the goal space is the space of
outcomes the agent actually produces, and when goals admit a meaningful
distance. A 3D position gives both: any piece position is a valid relabelled
goal, and Euclidean distance is a natural success predicate (`distance <
goal_tolerance`).

An occupancy grid fails on both counts. It is discrete and mostly invariant
under small motions, so almost every relabelled goal is identical to the
original and HER produces no learning signal; and grid Hamming distance says
nothing useful about how close a manipulation came to succeeding.

### Consequences

- `achieved_goal` and `desired_goal` must share **identical** bounds, since HER
  writes achieved values into the desired slot. The env asserts this via
  `test_goal_space_contains_achievable_positions`.
- Those bounds are deliberately wider than the board (a generous +/-5 m box), so
  a piece that gets knocked onto the floor still yields an in-space relabelled
  goal rather than an out-of-bounds observation.
- Only **one** piece is tracked. Multi-piece tasks in P2 will need either one
  goal vector per tracked piece or a per-episode choice of which piece is
  tracked; this decision does not settle that.
- Board-state reasoning (legality, whose square is whose) is deliberately *not*
  in the goal. If a task class needs it, it belongs in the observation or the
  reward, not the goal vector.
- **`is_success` is not a pure function of the goal pair.** Success also requires
  the piece to be resting on the board and the gripper to be open (see the
  amendment below), and gripper state is not part of `achieved_goal`.
  `compute_reward` remains a pure function of the goals, so HER relabelling
  itself is unaffected — but a relabelled transition's success flag cannot be
  recomputed from goals alone. Anything that needs per-transition success after
  relabelling must recompute it from stored state, not from the goals.

### Amendment, 2026-08-16: success is placement, not proximity

The original criterion was a single 3D distance test against `desired_goal`.
That was wrong: with a 30 mm tolerance, a piece still clamped in the gripper
25 mm above its target square scored as a success. The scripted expert
(`scripts/scripted_expert.py`) hit this in **all 50 of 50** episodes — every one
"succeeded" in mid-air and never ran its release or retreat waypoints.

Success now requires three independent conditions, all reported separately in
`info` so a failure says which one was missing:

| condition | test |
| --- | --- |
| `piece_on_square` | XY distance to the goal < `goal_tolerance` (30 mm) |
| `piece_resting` | piece origin within `rest_tolerance` (10 mm) of the board surface |
| `gripper_released` | gripper driver joint < `gripper_open_threshold` (0.2 rad) |

The gripper threshold is set from measurement, not guessed: the driver joint
reads 0.003 rad fully open, 0.542 gripping a 30 mm piece and 0.630 closed on
nothing, so 0.2 sits clear of any aperture that could still hold a piece.

Note this criterion has no *velocity* gate, so success can fire during the
settling wobble just after release. Adding one is a live option if placement
turns out to be scored while a piece is still toppling.

---

## D4 — Training-budget implication: image-observation configs need a wall-clock revision before P4

**Date:** 2026-08-24
**Status:** flagged, not resolved — needs a decision before P4 commits to a run schedule
**Evidence:** `scripts/profile_env.py`, full grid in `profiling/p1_throughput.csv`,
narrative in `profiling/README.md`

### Finding

Measured locally (WSL2, `MUJOCO_GL=osmesa`, software rasterization), image
observation configs run **100-400x slower** than state-only:

| observation mode | best measured steps/s | 500k-step projection |
| --- | ---: | ---: |
| state-only | 1420 (16 workers) | 0.10 h |
| state + 1 camera | 10.1 (16 workers, 64x64) | 13.8 h |
| state + 2 cameras | 5.6 (8 workers, 64x64) | 24.8 h |

The 500k figure is the pilot-matrix budget mentioned in the roadmap. These are
env-throughput-only projections (no policy-update cost included), so they are
a *lower bound* on wall-clock, not a training-time estimate.

### Why this matters

If the proposal's pilot-matrix wall-clock budget was built on an assumption
close to **~1 ms/step** (1000 steps/s) for every observation condition — which
is roughly what state-only delivers here — that assumption does not hold for
any image-observation config. At the measured local rates, a single
image-observation pilot run costs **13-25 hours** of env time alone, not
minutes. **Any image-config wall-clock budget in the proposal needs revision
before P4 locks in a run schedule** — either the per-run budget grows by
roughly two orders of magnitude, or the pilot matrix drops/reduces the
image-observation conditions, or both. This is a planning input, not something
this repo can resolve unilaterally.

### Why the rate is what it is

Direct measurement of `mujoco.Renderer.render()` in isolation found the cost
is dominated by the scene's 185,489 mesh faces (mostly the UR5e's visual
meshes) and is **nearly independent of resolution** — 64x64 and 128x128 cost
the same. OSMesa's software rasterizer is spending its time on geometry
processing, not pixel fill, at these sizes. See `profiling/README.md` for the
full breakdown.

### What these numbers are NOT

**These are a local OSMesa floor, not a cluster prediction.** `MUJOCO_GL=egl`
was tried on this machine and was not faster (WSL2 has no usable GPU
passthrough for rendering here, and EGL contexts actively crashed under
multi-worker `SubprocVecEnv`). A GPU-backed EGL renderer on real cluster
hardware should be substantially faster per frame, and the resolution-
independence seen here is a geometry-bound artifact that may not hold once
rendering is fill-bound on a GPU. **Re-run `scripts/profile_env.py` on the
cluster before finalizing any image-config wall-clock budget** — do not plan
against the numbers above.

### What does travel

The state-only number (100-400x faster, no rendering path at all) is backend-
independent and not subject to the caveat above.

This is one of Gate M1's exit criteria ("training budget revised in
writing") -- see `ROADMAP.md` for the full gate definition and status.

---

## D5 — HER decision, locked for T1: state-vector observations only

**Date:** 2026-08-25
**Status:** accepted (not a new decision -- writing down a constraint that
was already the working assumption, same posture as D3's original entry)
**Implemented by:** `robochess/tasks/t1_pick_and_place.py`

### Decision

T1's HER-compatible configuration is state-vector observations only.
RoboChessEnv has no image observation mode at all today, so there is
currently nothing to build or omit -- this entry exists so the constraint is
on record before P2/P3 add one, per the memory-wall arithmetic in the
roadmap: two 128x128x3 uint8 camera views at a 1M-transition buffer is
roughly 196 GB, not feasible. When an image observation mode is added, it
must not be paired with `HerReplayBuffer` under the current buffer sizing;
either keep HER to state-vector configs or drop to a much smaller,
lower-resolution image buffer (~100k transitions, 64x64, single camera) and
re-derive the memory budget explicitly before using it.

### Consequence for reward wrappers

`ShapedPickAndPlaceReward`'s gripper-approach term reads the live pinch-site
position, which is not recoverable from `(achieved_goal, desired_goal)`
alone, so its `compute_reward` is deliberately left unoverridden (falls back
to the base env's goal-only placeholder if HER is attempted with it anyway).
Only `SparsePickAndPlaceReward` is HER-safe, matching the proposal's
factorial design where HER is paired with the sparse arm.

---

## D6 — Grasp reliability degrades on the far half of the board; not caused by T1's wrapping, not yet explained

**Date:** 2026-08-25
**Status:** flagged, not resolved — needs a decision before P2 relies on
full-board random square sampling for training
**Evidence:** `scripts/scripted_expert.py --task t1`, full 64-square scan
(script run ad hoc, not checked in — reproducible via the snippet below)

### Finding

Running the scripted expert through `T1PickAndPlaceTask` with squares
sampled uniformly over all 64 reachable squares (per D2) gives **32/50
(64%) success**, against P1's 100/50 (100%) on its single fixed square (e4).
Restricting the *start* square and re-running singly across all 64 squares
(target held at e4, seed=0) shows the failures are not random:

```
     a  b  c  d  e  f  g  h
  8  ..  XX  XX  XX  ..  XX  ..  OK
  7  XX  ..  ..  XX  ..  ..  ..  ..
  6  ..  ..  ..  ..  ..  ..  ..  ..
  5  OK  OK  ..  ..  ..  OK  OK  OK
  4  ..  OK  OK  OK  OK  ..  OK  OK
  3  OK  OK  OK  OK  OK  OK  OK  OK
  2  OK  OK  OK  OK  OK  OK  OK  OK
  1  OK  OK  OK  ..  OK  OK  OK  OK
```
(OK = success, XX = `piece_off_board` during the `close` waypoint -- the
piece is flung 400-3800 mm off the table while the arm is holding still and
only the gripper is closing -- `..` = reaches the goal sequence but doesn't
land the placement criteria, not a catastrophic failure)

Ranks 1-4 (the half of the board nearer the arm base, which is where P1's
e4 sits) are almost universally reliable. Ranks 5-8 (the far half) are
mostly degraded or catastrophically unreliable, with the single exception of
h8, which succeeds cleanly despite being the single most extended square in
the whole 64-square set per D2's own sweep.

### What this is NOT

- **Not a T1PickAndPlaceTask bug.** The failure is 100% determined by the
  *start* square alone -- holding start fixed and varying the target has no
  effect (`h8 -> h7`, `h8 -> e4` both succeed; `d8 -> b3`, `d8 -> h1` both
  fail the same way). T1's teleport/goal-set/info-augmentation code is
  identical regardless of which square is involved.
- **Not a contradiction of D2.** D2 swept static top-down IK convergence,
  which still holds -- `solve_ik` converges for every failing square here too
  (zero `ik_failure` outcomes in either run). This is a *dynamics/grasp*
  problem that only shows up when the waypoint sequence actually executes
  contact-rich manipulation, not a kinematic reachability problem. D2's
  reach-envelope numbers are unaffected.

### What it might be (not confirmed)

The `close` waypoint holds the arm still and only closes the gripper
(`ctrl=255`) -- the piece is flung away during that hold, not during any
arm motion. The leading hypothesis is that arm configurations required to
reach the far half of the board need larger held-position corrective
torques (less favourable moment arms against gravity than the
near-central e4 configuration the scripted expert's heights were tuned
against), and closing the gripper on top of that injects enough contact-
solver instability to launch the piece. This is unconfirmed -- it was not
isolated further (e.g. by comparing a direct-teleport-to-grasp-pose against
the waypoint-driven approach) before this was written down.

### Why this might not matter for training

The scripted expert is a fixed, open-loop, joint-space controller tuned
once against a single square; an RL policy uses closed-loop, continuous
control and never blindly repeats that exact tuned trajectory, so it may
not hit the same failure mode at all. But it's also possible the underlying
cause is a genuine simulation robustness issue at more-extended arm poses
that would affect any controller, scripted or learned. Nothing in this repo
resolves which of those is true.

### Consequence

`T1PickAndPlaceTask`'s default `reachable_squares` was left at all 64
squares (`REACHABLE_SQUARES = ALL_SQUARES`) rather than narrowed to the
reliable near-half, since narrowing it unilaterally would silently shrink
the task class's scope without sign-off, and D2's kinematic reachability
claim is still correct for all 64. But this means uniform random square
sampling -- T1's default reset behaviour -- will keep drawing squares from
the unreliable far half during training. **Needs a decision before the P2
Gate M2 training runs and before the pilot matrix (P3):** narrow
`reachable_squares` to the validated-reliable subset for now, spend time
root-causing the dynamics issue directly, or accept it and treat any
resulting depressed success rate on far-half episodes as expected until
this is resolved.
