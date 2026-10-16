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
| `piece_resting` | piece origin within `PLACEMENT_REST_TOLERANCE` (5 mm, F2) of the board surface. `rest_tolerance` stays 10 mm for T1's `lifted` and D18's grasp check |
| `piece_still` | (F2) piece linear speed < 0.01 m/s for 3 consecutive steps |
| `piece_upright` | (F2) piece tilt from vertical < 10 degrees for 3 consecutive steps |
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

## D7 — Grasp reliability degrades on the far half of the board — RESOLVED: no gravity compensation on the UR5e's position actuators

**Date:** 2026-08-25 (diagnosis); **RESOLVED 2026-08-27, commit `8139b27`**
**Status:** resolved. Root cause confirmed: steady-state PD tracking error
from uncompensated gravity load on the arm's position actuators, which also
required correcting `GRASP_Z`'s grasp-geometry rationale (see "Resolution"
below). The residual ~8-16% failure rate remaining on the broad 64-square
scan is a *separate*, already-diagnosed mechanism — see D8 — not a
recurrence of this one.
**Evidence:** `scripts/scripted_expert.py --task t1`, full 64-square scan
(script run ad hoc, not checked in — reproducible via the snippet below);
instrumented diagnostics in `scripts/scripted_expert.py` (`run_episode(log=...)`)

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

### Diagnostic passes 1-3 (2026-08-25): what's confirmed, ruled out, and still open

Three follow-up diagnostic passes instrumented `scripts/scripted_expert.py`
(per-timestep joint state/torque, end-effector tilt, per-pad contact
force, and a `frame_skip=1` probe for sub-macro-step contact timing) and
compared e4 (reliable) against the far-board failures. This closes out the
diagnostic effort by decision -- not because the mechanism is now fully
understood, but because the returns on further isolation were diminishing
relative to the time spent. **Mitigation design (what T1 actually does
about this) is separate, upcoming work, not attempted here.**

**Ruled out, with evidence:**
- *Wrist tilt.* Present (1-2° larger on far squares than on e4) but small
  in absolute terms, and doesn't spike at the failure moment -- if anything
  it *decreases* as the piece is flung. Not the trigger.
- *Joint-limit proximity.* The opposite of causal: catastrophic squares sit
  *farther* from every joint limit (~2.0 rad margin) than the reliable e4
  (~0.92 rad margin at the elbow). Ruled out.
- *Singular configuration.* Checked via Jacobian condition number at the
  pinch site: elevated on far squares (~11 vs e4's ~7.7) but nowhere near
  singular (smallest singular value stays >=0.17 throughout), and flat
  across the whole `close` phase -- can't explain *when* the piece gets
  flung. Ruled out as the trigger.
- *End-effector yaw / orientation drift.* Essentially zero (<0.02°) and
  identical between paired squares. The top-down grasp fully specifies a
  6D pose for a 6-DOF arm, so yaw is pinned by the IK target, not free.
  Ruled out.
- *IK path-dependence.* `solve_ik` seeds from the arm's live `qpos`
  (path-dependent by construction), but empirically, solving the same
  target from the home keyframe vs. from a real episode's post-descend
  state converged to the same joint solution (<0.09° apart) for the one
  case tested. Doesn't explain the asymmetry, though only one case was checked.
- *Contact-timing offset alone.* Pads contact 4-18ms apart, not
  simultaneously -- but the offset's direction and magnitude are similar
  for both members of a mirror pair regardless of outcome (e.g. d8 and e8
  both show the right pad touching first, 6ms and 10ms respectively; one
  goes catastrophic, the other doesn't). Timing offset alone doesn't
  discriminate.
- *Peak tangential force as a simple threshold.* Tabulated peak tangential
  pad force across all 10 diagnosed squares (e4, c8, and the four mirror
  pairs): d8 (catastrophic) has the *lowest* peak of the entire set
  (23.2N), below every partial (30-59N) and every success (31-34N); h7
  (partial) sits above it at 30.0N, essentially level with h8 (success,
  31.4N). No magnitude cleanly separates outcomes. One outlier (b8, 715N)
  suggests the mechanism can run away once triggered, but the other two
  catastrophic squares don't show that same extreme, so this doesn't
  rescue a threshold reading of the set.

**Confirmed / correlated, not fully explanatory:**
- *Tangential (shear) contact-force imbalance at first pad contact* is the
  most timing-precise anomaly found: normal forces stay roughly balanced at
  first touch in every case, but tangential forces are heavily one-sided
  (ratios up to 16:1) in every catastrophic square and mild in every
  success, including the far-but-reliable h8 -- ruling out "far squares
  always show this."
- *The second pad to make contact tends to bear the dominant tangential
  force* in catastrophic cases: true in 3/3 tested (d8, b8, a7). Not
  sufficient on its own -- h7 shows the identical signature and only
  reaches a partial, not catastrophic, outcome.
- *Reach distance* correlates moderately with severity across the full
  64-square grid (Pearson r=+0.72) but leaves roughly half the variance
  unexplained, and individual joint angles at the grasp pose (elbow,
  shoulder_lift, wrist_1) correlate about as strongly as distance and no
  more -- expected, since they're kinematically coupled to distance, not
  an independent factor.
- *A left/right handedness asymmetry* shows up in mirror-pair outcomes
  (same distance, opposite-sign azimuth, different outcome in 4/5 pairs
  tested, files a-d consistently worse than their e-h mirror) that
  distance or joint-angle *magnitude* cannot produce by themselves, since
  those are identical for a mirror pair. Points at something with a
  definite sign/handedness -- solver branch selection, or a real asymmetry
  in the model -- but which one is unconfirmed.

**Genuinely unexplained, left open:**
- *Why* the tangential contact becomes asymmetric in the first place on
  far squares -- this is upstream of everything measured and was not
  isolated.
- *What determines escalation.* The second-pad-dominant signature appears
  in both catastrophic and merely-partial cases (h7 vs. d8/b8/a7); nothing
  found here explains why some instances run away and others don't.
- *The origin of the handedness asymmetry itself* (solver artifact vs.
  model geometry) was not distinguished.

### Why this might not matter for training (superseded, see Resolution below)

This subsection reflects the state of understanding as of 2026-08-25, when
the mechanism was still unconfirmed: the scripted expert is a fixed,
open-loop, joint-space controller tuned once against a single square, and
it was genuinely unknown whether a closed-loop RL policy would hit the same
failure mode or not. **This is no longer an open question about *whether*
the cause is controller-specific or a general simulation issue** — it is
confirmed as the latter (uncompensated gravity load on the arm's own
actuators, a property of the model, not the scripted expert's control
scheme). What remains open, and is now P3's job to check rather than
speculate about, is narrower: whether an RL policy's continuous per-step
feedback lets it compensate for this tracking error on its own during
training, the way the scripted expert had to be given an explicit fix to
do. See "Resolution" below and the Consequence section that follows it.

### Resolution (2026-08-27, commit `8139b27`)

**Root cause, confirmed with physical/geometric evidence, not just
correlation:** the UR5e's position actuators have no gravity compensation
(`gainprm` 2000/500, `gravcomp` unset, confirmed inherited unchanged from
`mujoco_menagerie`), so holding any joint target against gravity leaves a
**steady-state** PD tracking error — a static equilibrium, not a slow
transient. Measured 1.54 deg at `shoulder_lift` on the far rank, against
1.615 deg predicted directly from `qfrc_bias/kp`. In task space this is
~27 mm of vertical sag and ~12 mm of horizontal under-reach at rank 8 —
enough to drop the grasp off its intended contact band and push the pinch
point past the gripper pad's ~11.00 mm half-extent, which is what separated
catastrophic squares from surviving ones across the 18 squares diagnosed in
"Diagnostic passes 1-3" above. That session's own findings (the tangential
contact-force imbalance, the second-pad-dominant signature, the left/right
handedness asymmetry) are consequences of this one cause — the piece being
contacted at the wrong height/offset before closing ever starts — not
independent clues pointing at separate mechanisms.

**Four fix attempts, tried in order, each failed for a confirmed and
distinct reason — do not retry these in this form:**

1. **Uniform closing-velocity ramp.** Broke the previously-reliable e4
   baseline: slowing the close phase let the gripper settle into a
   shallower, weaker equilibrium than the abrupt close it replaced.
2. **Contact-triggered dwell.** Preserved e4, but had zero effect on the
   four originally-catastrophic squares — the piece is already
   mispositioned before closing starts, so waiting longer after contact
   doesn't help.
3. **Retargeting `solve_ik()` from the attachment site to the pinch site.**
   Changed the pre-close offset by <0.1 mm on every square tested. IK
   converges to essentially the same pose regardless of which site it
   targets, so the residual was never coming from *what point* IK aimed at
   — it is downstream of IK, in physical tracking under load.
4. **Tightening `settle_tol` from 0.02 to 0.002 rad.** Failed its own
   verification: every waypoint on every square exhausted its full step
   budget without ever reading "settled," and the catastrophic squares
   stayed catastrophic even at full budget — proof the residual is a floor
   (steady-state tracking error), not something more time resolves.

**The fix that worked: task-space residual compensation, `descend` only.**
After driving to a waypoint, `_run_waypoint()` measures where the pinch
site actually ended up, feeds the shortfall back into the IK target,
re-solves, and re-drives — capped at `MAX_RESIDUAL_ITERS = 2` (converges to
1-2 mm in practice, since the error is a stable offset, not something that
oscillates). `close` and `release` reuse the pose their predecessor
converged to rather than re-solving back to the uncompensated target.
Compensating `place` as well was tried and rejected: by the time `place`
runs, the piece is already in contact with the board, so the measured
"shortfall" partly reflects the board pushing back rather than free-space
tracking error, and the correction loop fights that contact — at larger
correction budgets it drove the piece off the board entirely. See
`COMPENSATED_WAYPOINTS = frozenset({"descend"})` in `scripts/scripted_expert.py`.

**`GRASP_Z` recalibrated 0.8035 -> 0.79925, and the original P1 rationale
corrected.** The original value was tuned by outcome while the arm still
had this uncorrected sag, so 0.8035 was never actually commanding the
"stem above centre-of-mass" grasp P1's comment claimed — sag was doing the
calibration by accident (~5.6 mm at e4), landing the true contact on the
piece's head regardless of what anyone intended. Investigating this
directly (not just recalibrating and moving on) found:

- A pure stem-only grasp is **geometrically impossible**: the piece's head
  radius (9 mm) exceeds its stem radius (8 mm), and the gripper pad's own
  vertical extent is taller than the stem band, so no height exists that
  contacts only the stem.
- A "form-closure" straddle grasp (base below + head above) was proposed as
  the safety criterion to replace "above COM," but **failed direct
  measurement**: at the geometric window's centre, tilt during lift
  reproduced P1's original tipping failure almost exactly (68-71 deg vs.
  P1's ~58 deg), aperture converged short (0.542 rad, not ~0.65), and
  contacts were base-only, not base+head.
- Root mechanism: gripper closing is **"first significant contact wins,"
  not a static overlap calculation** — the controller stops the instant it
  meets resistance from whichever feature enters the pad's range first. A
  height overlapping both the base and a narrower feature above it
  converges to a shallow base-only grip, not a deeper grip past it. A
  sweep confirmed a sharp discontinuity in converged aperture (0.647 at one
  height, 0.54 barely 1 mm away) that a static geometry calculation could
  not have predicted.
- **Final validated value, `GRASP_Z = 0.79925`:** clears the base with real
  margin and grips the piece's **head only**. Validated directly across an
  18-square sample spanning 43-99% of rated reach, all four quadrants,
  multiple mirror pairs: tilt stays low through the lift, aperture
  converges to 0.645-0.647 uniformly (matching the original ~0.65 target).

**Validation:** 18/18 succeed with no regressions on the previously-working
squares; all 6 previously-catastrophic squares (a7, b8, c8, d8, e8, a8) now
succeed. Full 64-square scan (target e5): 51/63, as measured and recorded
at commit time. CLI regression, 50 random-square episodes: 46/50 (92%), up
from the 64% historical baseline, with no catastrophic `piece_off_board`
failures remaining. **The residual failures are a separate, already-
diagnosed mechanism — see D8 — not a recurrence of this one.**

(A later re-run of the same 64-square scan, during D8's diagnostic pass on
2026-08-27, measured 53/63 on identical code — 2 squares better than the
51/63 recorded above. This is expected, not a discrepancy in D7's fix
itself: the extra 2 squares are release-phase outcomes, D8's chaotic
mechanism, sensitive to run-to-run environment numerics. D7's own claims —
the 18/18 grasp-reliability sample and the elimination of the 6
catastrophic squares — do not depend on that number and are unaffected.
See D8's own note on environment sensitivity.)

### Consequence (superseded the "needs a decision" framing below as of the Resolution above)

With D7 resolved, `T1PickAndPlaceTask`'s default `reachable_squares` stays
at all 64 squares (`REACHABLE_SQUARES = ALL_SQUARES`) — narrowing it is no
longer the live question D7 originally posed, since the fix restores
reliability across the reach envelope rather than only on a near half.
D2's kinematic reachability claim was correct for all 64 squares throughout
and needed no revisiting.

The question this resolution hands to P3 instead: **does the real RL
action space for T1-T5 drive the same uncompensated position actuators the
scripted expert needed explicit compensation for**, meaning a trained
policy could inherit the same gravity-droop tracking error. RL policies get
continuous per-step joint feedback (unlike the scripted expert's open-loop
waypoints), so may compensate for this on their own during training — but
that needs checking directly before P3's pilot matrix, not assumed.

---

## D8 — Release-induced rotational instability on the head-only grasp; universal across passes and fails, chaotic in outcome

**Date:** 2026-08-27 (diagnosis); **RESOLVED 2026-08-27** (fix-design
session, Candidate 1 -- see "Resolution" below).
**Evidence:** `scripts/scripted_expert.py --task t1`, driven through a
diagnostic subclass (ad hoc, not checked in) that extends `LOGGED_PHASES`
to the post-grasp waypoints (`lift`, `traverse`, `place`, `release`,
`retreat`) and logs the pinch site's own position directly — D7's logging
stopped at `close` and used the attachment site, not the pinch point,
because D7's mechanism was fully upstream of the grasp.

### Finding

The full 64-square scan (target fixed at e5, every other square as start)
that validated D7's fix surfaces a second, separate failure population,
downstream of a clean grasp: **10 of 63 squares fail** (b5, c1, e2, e6, f1,
f3, f6, g6, g7, h6; 53/63 succeed, as measured directly in this session —
see the discrepancy note below on why this differs from an earlier
snapshot).

Per-waypoint tracing (piece position, gripper aperture, pad-piece contact
count, and pinch-to-piece distance at every step of every waypoint, for all
10 failing squares) shows:

- Grip is continuous through `lift`, `traverse`, and `place` for all 10 —
  pad-piece contact stays nonzero (6-8) throughout, and piece-to-pinch
  offset barely moves.
- At the end of `place`, every failing square has the piece within 5-11 mm
  of the goal — well inside the 30 mm `goal_tolerance`. Placement accuracy
  itself is not the problem.
- The divergence starts precisely at `release`: pad-piece contact drops to
  0 within 0-1 steps of the gripper beginning to open, and the piece's
  angular speed spikes violently — 12.8-23 rad/s across the 10 failures.

### What this rules out

**Not an uncompensated place/release tracking-error problem, and not the
same family as D7.** `place`/`release` genuinely receive no task-space
compensation (`COMPENSATED_WAYPOINTS = frozenset({"descend"})`, confirmed
both from the code and from `compensation_iters` in the run record), but
the place waypoint's own tracking error is small and is not the source of
the eventual 20-103 mm final errors — the corruption happens entirely
after a good placement, during release itself. This was the leading
candidate mechanism going into this diagnostic pass; the data does not
support it.

**Not reach-dependent.** Pearson r(reach, fail) = -0.06 across all 63
squares, confirming the roadmap's own earlier note and ruling out a
D7-style, reach-scaling mechanism.

### Mechanism

One shared cause, not two, despite the bimodal final-distance groups:

- **Group A** (b5, c1, e6, f6): tumbles at release, then settles flat back
  onto the board — just shifted sideways, landing 30-33 mm from target.
- **Group B** (e2, f1, f3, g6, g7, h6): lands in a metastable
  perched/edge-balanced state, still spinning, piece height stuck 15-19 mm
  above the board (five of six at *exactly* 0.7809 m — consistent with
  balancing on a specific geometric edge of the piece's base) and never
  settles within the episode's remaining step budget.

Both are outcomes of the same event: the release waypoint commands an
abrupt, uncushioned transition from fully closed (`ctrl=255`) to fully open
(`ctrl=0`) on a grasp that D7 deliberately committed to gripping the
piece's **head only** — a rotationally underconstrained, near-spherical
contact. The instant compressive force drops, the piece is free to spin.

Critically, **this spin is universal, not specific to the squares that
fail.** A sample of 12 passing squares showed the same peak
release-angular-speed range (mean 13.65 rad/s, up to 30.34 rad/s on a1) as
the 10 failures (mean 16.65, range 12.78-23.05). Every T1 episode using
this grasp geometry undergoes a violent release event; passing squares are
landing inside tolerance essentially by chance from a chaotic process, not
because release is reliable there. **The current ~84% (53/63) pass rate on
this scan is sitting on a chaotic edge, not a robust margin.**

This is the flip side of D7's own fix, not an independent bug:
`scripts/scripted_expert.py`'s comment on `GRASP_Z` already warned, before
D7's recalibration, that "a pure head grasp is a single sphere and simply
rotates in the fingers (measured: piece placed 38 mm off target)." D7
committed to exactly that grasp geometry to solve the pre-grasp reliability
problem, and this is that warning materializing on the release side.

### Checked and ruled out as a clean discriminator

Neither wrist orientation at grasp (`wrist_3 mod 180`, checked across both
outcome groups) nor peak release angular-speed magnitude cleanly separates
successes from failures, or Group A from Group B — ranges overlap heavily
in both cases. This mirrors D7's own "first significant contact wins"
finding: a chaotic threshold effect, not a variable with a clean cutoff. No
discriminating variable was found despite checking.

### On the 12-vs-10 discrepancy with the external roadmap doc

An earlier snapshot (recorded outside this repo, in the project's roadmap
doc) reported 12 failing squares on this same scan, including four (c6,
d6, c7, f7) that this session's runs do not reproduce, and missing two
(f6, g7) that this session's runs do. Two independent runs in this
environment reproduced the identical 10-square failure set both times, so
this is not run-to-run noise within a single environment. Given the
mechanism identified above — a chaotic process sensitive to the exact
release dynamics — small numeric differences between environments (BLAS
backend, MuJoCo point version) are expected to tip a handful of borderline
squares between outcomes. **This is consistent with, not contrary to, the
diagnosis:** the specific failing-square list is not a stable fact about
the environment; only the existence and rate of this failure mode is. Treat
any specific square list (this one included) as a snapshot of one
environment, not a fixed property of the task. D7's own 51/63-vs-53/63
discrepancy (see D7's Resolution section) is the same phenomenon.

### Consequence (superseded — see "Resolution" below)

Not attempted at diagnosis time, by the project's diagnose-then-fix
convention. Candidate directions floated for a future fix-design session:
damping the release open-command (a velocity ramp, analogous to D7's
rejected closing-side attempt, though D7's attempt failed for the closing
case for reasons that may or may not transfer to opening); a grasp geometry
with more rotational constraint than a pure head contact (revisits the same
base/stem/head tradeoff D7 already investigated); or a settle-and-check
step after release before declaring the waypoint sequence finished, so a
piece caught mid-spin gets more time rather than being scored against
whatever state the episode happens to end in. The first and third of these
are what the Resolution below actually combines.

### Resolution (2026-08-27, fix-design session)

**Candidate 1 adopted: ramped release + piece-settle gate.**
`scripts/scripted_expert.py`'s `release` waypoint no longer commands the
open ctrl in one jump; it ramps linearly from fully closed to fully open
over `RELEASE_RAMP_MACRO_STEPS = 15` macro-steps (`_release_ramp`), and the
waypoint's own completion is gated on ctrl having reached its final open
value AND the piece's angular speed staying below `PIECE_ANG_SETTLE_TOL`
(0.05 rad/s) for `SETTLE_CONSECUTIVE_STEPS` (3) consecutive macro-steps
counted from that point forward — not on the arm/gripper settle check every
other waypoint uses. `release`'s own budget was raised 35 -> 120 macro-steps
to give this room to run (measured up to 69 macro-steps needed on one
square in validation). The mechanism: a gradual force reduction gives the
piece's board contact time to damp out uneven pad-loading before full
separation, instead of removing it all at once while spin is still
building.

**This was validated, not assumed, against a specific alternative
hypothesis: that the ramp's apparent fix was really a time-budget
artifact, or worse, a success-check timing artifact (see D9).** A three-way
paired comparison on 20 squares — (A) instant release, (B) the ramp, (C)
instant release plus a no-op dwell padded to match (B)'s own total
macro-step count exactly — found C behaved like A (70% success), not B
(100%), even though C had at least as much total time as B on every square.
This ruled out "more time alone" as the explanation. It also directly
surfaced D9 (is_success had no piece-settle requirement, so B's early
100% figure was partly an artifact of checking success while the ramp was
still mid-transition) — D9 was fixed first, and the ramp was re-validated
against the corrected, now-production `is_success`, not the criterion that
produced the original misleading 100%.

**Full-board validation (Section 5 Part D), corrected `is_success`
throughout, 3 fixed targets (e5, a1, h8) x 63 start squares each, 350-step
episode budget:**
- (ii) old instant release: 149/189 (78.8%)
- (iii) Candidate 1: **187/189 (98.9%)**, zero regressions relative to (ii)
  on any square across all three targets.

Real CLI regression (`scripts/scripted_expert.py --task t1 --episodes 50
--max-episode-steps 350`, default random start/target sampling — the actual
suite D7 originally reported 46/50 against): **50/50 (100%)**, confirming
no D7 regression from either the release-ramp change or the `is_success`
piece-settle requirement.

**Residual, out of scope:** 2 of the 189 full-board episodes still fail
(c2->a1, h1->h8), both `piece_off_board` during `traverse`/`place` --
*before release even begins* -- present under both (ii) and (iii)
identically, so not a Candidate-1 regression. Later measurement showed these
are two *different* unrelated mechanisms, not one: h1->h8 is a geometric
excursion with the grip intact (**D10**) and c2->a1 is a contact-solver force
spike and ejection (**D11**). Both are logged separately.

**Single-environment-build caveat:** all of the above (the A/B/C
comparison, the full-board validation, the CLI regression) was run in one
process, on one MuJoCo/BLAS build, within one session. D7 and D8's own
diagnostic history (the 51/63-vs-53/63 discrepancy, the 12-vs-10
failing-square-list discrepancy) already established that this specific
system is sensitive to small numeric differences between environments
(BLAS backend, MuJoCo point version) — enough to tip a handful of
borderline squares between outcomes. The 98.9%/100% figures above should be
read as "this fix is a large, real improvement, validated at full-board
scale," not as "exactly 187/189 and 50/50 are stable constants that will
reproduce bit-for-bit on a different machine or a future MuJoCo version."
Re-validating on the actual training/cluster environment before leaning on
these exact numbers (e.g. before finalising P3's pilot matrix) is still
worth doing, per the same standing lesson.

---

## D9 — `is_success` has no piece-velocity/settle requirement; can register success mid-spin

**Date:** 2026-08-27
**Status:** RESOLVED, fixed at the source in `robochess_env.py`.

**Finding:** the production `_is_success()`/`_build_info()` criterion
(`on_square` AND `resting` AND `released`) has no requirement that the piece
itself be at rest -- no velocity or settle-streak check on the piece's
linear/angular velocity, unlike the analogous arm/gripper settle-check
(`SETTLE_CONSECUTIVE_STEPS = 3` in `scripts/scripted_expert.py`), which does
require multiple consecutive low-velocity readings before treating a
waypoint as settled. Traced at per-physics-substep resolution across
several squares' unmodified (instant) release, `is_success` fires while the
piece still has substantial angular velocity in multiple cases -- observed
up to 17.7 rad/s (f6), 13.16 rad/s (e2), 7.82 rad/s (c1), 5.52 rad/s (g7),
4.91 rad/s (f1) -- not at rest by any reasonable standard.

**Consequence for existing numbers:** every D7 and D8 success/failure count
to date (18/18, 46/50 CLI regression, 51-53/63 full scan) was measured
against this same under-specified criterion, so in principle some recorded
"successes" could be catching the piece mid-transient rather than
genuinely settled. A direct recheck was run: the same 20-square sample used
in D8's release-instrumentation work (the 10 originally-failing squares
plus 10 chosen passing squares) was re-evaluated under a corrected
criterion -- requiring ctrl to have reached its fully-open target, plus the
piece's angular speed below 0.05 rad/s for 3 consecutive macro-steps
counted from that point forward, not from episode/phase start -- and **no
square's pass/fail outcome flipped**. The existing rates therefore likely
hold approximately, but this has only been confirmed at n=20, not at the
original 50-63 square sample sizes, so they remain formally unverified at
those larger Ns.

**Standing rule:** any future D8 fix must be validated against this
corrected (settled) success criterion, not the current production one -- a
fix that only looks good under the current criterion could be passing
because it moves *when* success is checked (catching a calmer instant)
rather than because it genuinely resolves the release-induced instability.
This was directly observed during D8's fix-design work: an initial 15-step
gripper-open ramp appeared to fix 100% of a small sample under the
production criterion, but tracing revealed success was registering while
ctrl was still mid-ramp (not yet at its open target) in every case -- see
D8's fix-design session log for the full account, including the corrected
re-run that confirmed the ramp's benefit holds even under the tightened
criterion.

### Resolution (2026-08-27)

**Option chosen: add the missing piece-settle requirement directly to
`RoboChessEnv._is_success()`/`_build_info()`** (the alternative considered
was a lighter single-step velocity check with no consecutive-streak
requirement, which would have avoided adding new per-episode state but
reintroduces the exact "a single instantaneous reading can be a transient"
risk this project already fixed once for the arm/gripper settle-check --
rejected for the same reason).

`_success_terms()` now returns a fourth condition, `settled`: the piece's
own angular speed has stayed below `PIECE_ANG_SETTLE_TOL` (0.05 rad/s -- the
same value established throughout this investigation, not re-derived) for
`PIECE_SETTLE_CONSECUTIVE_STEPS` (3) consecutive `step()` calls, counted
continuously rather than gated by the other three conditions. This required
new persistent state, `self._piece_ang_settled_streak`, since everything
`_is_success()` previously computed was a pure function of the current
physics state with no memory across steps. It is initialised in `__init__`
and explicitly reset to 0 in `reset()`, so it cannot leak between episodes
(a forgotten reset here would have silently corrupted the following
episode's early steps rather than failing loudly).

**Blast radius, checked before implementing:**
- `RoboChessEnv.step()`'s own `terminated` depends on `_is_success()`, so
  every consumer now takes longer to terminate on success, not only the
  scripted expert -- this was already anticipated from Section 5 Part A's
  step-budget investigation.
- `SparsePickAndPlaceReward.step()` uses `info["is_success"]` directly as
  the real (non-relabelled) training reward -- this change alters actual RL
  reward *value* timing for the sparse arm of the 2x2x2 study, not just
  diagnostics. `ShapedPickAndPlaceReward` was checked and is NOT affected in
  its reward *value* (it computes a continuous distance-based reward every
  step, independent of `is_success`); it only inherits the same later-
  termination timing every consumer gets.
- Four existing unit tests constructed a scenario, took exactly one
  `env.step()`, and asserted immediate success/termination -- these needed
  updating to drive `PIECE_SETTLE_CONSECUTIVE_STEPS - 1` steps first:
  `tests/test_env_basic.py::test_success_when_all_three_conditions_hold`,
  `tests/test_env_basic.py::test_terminated_and_truncated_are_mutually_exclusive`,
  `tests/test_t1_pick_and_place.py::test_success_when_all_three_conditions_hold`,
  and `tests/test_t1_pick_and_place.py::test_sparse_reward_is_exactly_zero_or_one`.
  The fourth of these was missed in the initial grep-based blast-radius scan
  (it doesn't reference `is_success`/`piece_resting`/etc. by name, only
  `reward`) and was only caught by actually running the full suite --
  worth remembering that a name-based blast-radius search is not a
  substitute for running the tests.

**Validation:** full test suite 48/48 passing after the fix and test
updates. Full-board validation (Section 5 Part D): three conditions across
3 fixed targets (e5, a1, h8) x 63 start squares each, corrected
(now-production) `is_success` throughout --
(ii) old instant release + 350-step budget: 149/189 (78.8%);
(iii) Candidate 1 (ramped release + piece-settle gate) + 350-step budget:
187/189 (98.9%), zero regressions relative to (ii) on any square across all
three targets. See D8's entry for the two residual failures (both
`piece_off_board` during `traverse`/`place`, before release even begins --
a different, out-of-scope mechanism, present under both conditions, not a
Candidate-1 regression).

---

## D10 — `piece_off_board` on h1->h8: joint-space interpolation carries the gripped piece outside the board footprint during `traverse`

**Date:** 2026-08-27 (logged); characterized 2026-09-10 (verification
session); root-caused 2026-09-10 (kinematic replay, same session); fix
implemented and measured 2026-09-10 (fix-design session)
**Status:** **root-caused; fix implemented, validated, and left DISABLED.**
The fix removes the excursion on both D10 pairs with ~75 mm of measured
margin, but enabling it reshuffles the marginal placement population that D11
describes, so it ships behind `TRAVERSE_TASK_SPACE = False` in
`scripts/scripted_expert.py` pending D11. See "The fix, and why it is off by
default" below. **D10 is not resolved.**

**Split note:** this entry and D11 were originally logged as a single D10
covering both residual full-board failures, on the assumption they shared a
mechanism ("long diagonal reaches"). Measurement showed they do not share
one — the two failures have nothing in common beyond both setting
`piece_off_board` — so they were split. See "Correction to the original
framing" below; D11 carries the c2->a1 case.

**Finding:** on h1->h8 the piece trips `_piece_off_board()` at global step
76, in `traverse`, at `phase_step` 2, on the `outside_xy` branch. The grip
is entirely secure at that moment and the piece's own dynamics are
unremarkable. It is carried out of the board's XY footprint by the arm; it
does not fall, slip, or come loose.

Board geometry: centre (0.0000, 0.5200), slab half-extent (0.3120, 0.3120),
plus the predicate's 0.5 * SQUARE_PITCH margin (0.0390) gives an XY bound of
0.3510.

```
step  phase      pstep  piece_xyz                   lin    ang   ncon  Lnorm  Rnorm
  73  lift          17  ( 0.2668, 0.2446, 0.8373)  0.01   0.00     6  44.61  44.62
  74  traverse       0  ( 0.2817, 0.2940, 0.8391)  2.60   0.53     6  44.57  45.58
  75  traverse       1  ( 0.3305, 0.3759, 0.8541)  2.17   0.41     6  44.35  44.79
  76  traverse       2  ( 0.3551, 0.4504, 0.8602)  1.77   0.32     6  44.45  44.73  <-- trips
```

At the trip: 6 pad-piece contacts, left/right normal forces balanced at
44.45 / 44.73 N (the ordinary steady-grip value), angular speed 0.32 rad/s,
linear speed 1.77 m/s, height 0.8602 (transit height, well clear of the
board). x = 0.3551 against the 0.3510 bound — **an excess of 4.05 mm**.

**The excursion is in the path, not in the piece.** h1 and h8 are on the
same file: h1 = (0.273, 0.247), h8 = (0.273, 0.793). A straight-line
traverse holds x = 0.273 and clears the bound by 78 mm. The executed path
bulges outward in +x over three macro-steps — 0.2678 (end of `lift`) ->
0.2817 -> 0.3305 -> 0.3551 — i.e. the arm swings an arc that leaves the
board footprint while the piece is still gripped and airborne.

**Root cause: the planned path leaves the footprint before physics is
involved.** `traverse` is commanded as a single joint-space target — the
expert solves IK for the traverse pose and `_drive_to()` holds that target
while the position actuators drive to it. Nothing constrains the *route*
between the lift pose and the traverse pose, and the forward kinematics of a
straight line in joint space between two widely-separated poses is a
Cartesian arc, not a Cartesian straight line.

Replaying that planned path kinematically — forward kinematics along the
joint-space interpolation between the two poses, with no physics stepped at
all — puts the carried piece **25-27 mm outside the 0.3510 m XY bound** at
peak. The executed path exceeds the bound by 4.05 mm.

So physics *reduces* the excursion by roughly 20 mm rather than causing it:
the arm never fully tracks the commanded corner of the arc, and the realized
path cuts inside the planned one. This rules the dynamics out from the other
direction as well as the trace above does — contact, grip and the piece's own
behaviour cannot be responsible for an overshoot that is already present, and
larger, in a trajectory that never touches the solver.

**Mechanism (confirmed):** joint-space interpolation between two
widely-separated poses swings the end effector, and the piece it carries,
outside the board footprint even when both endpoints sit well inside it. h1
and h8 are both at x = 0.273, 78 mm clear of the bound; the arc between them
is not. The ordered family below is largely this same arc taken shorter:
h2->h8, g1->h8 and h3->h8 are all traverses to h8 from the h and g files. The
one member of that list which is not, `a8->a1`, is the mirror geometry on the
a file — the other full-length corner-to-corner traverse in the sweep — and it
too sits among the closest approaches to the bound.

**Second instance: the reverse pair h8->h1.** The widened five-target sweep
also fails `h8->h1`, the same file traversed in the opposite direction, and
it is this entry's mechanism rather than D11's — measured, not inferred from
direction symmetry:

```
step  phase      pstep  piece_xyz                   lin    ang   ncon  Lnorm  Rnorm   exc_mm
 139  lift         24  ( 0.2686,  0.7845,  0.8116)  0.00   0.00     7  44.65  44.66   -82.41
 140  traverse      0  ( 0.2740,  0.7273,  0.8028)  2.20   1.11    10  44.25  46.21   -76.97
 141  traverse      1  ( 0.3174,  0.6417,  0.8212)  2.37   0.56     8  43.25  45.90   -33.61
 142  traverse      2  ( 0.3460,  0.5640,  0.8379)  1.85   0.37     8  44.09  45.15    -5.00
 143  traverse      3  ( 0.3554,  0.4992,  0.8471)  1.46   0.25     8  44.36  44.84    +4.36  <-- trips
```

Peak excursion **+4.36 mm** at `traverse` p3, with the grip completely
intact: 8 pad contacts at the trip and never below 7, left/right normal
forces 44.36 / 44.84 N (0.48 N imbalance, 2.7 N maximum across the whole
lift+traverse span), total pad force 89.2-90.5 N against D11's 229.8-504.1 N,
and piece angular speed 0.25 rad/s at the trip against D11's 27.6-39.1 rad/s
onsets. Every discriminator reads D10, none reads D11. x walks outward
0.2686 -> 0.2740 -> 0.3174 -> 0.3460 -> 0.3554 between two squares that both
sit at x = 0.273, with y inside bound throughout: the violated axis is x
alone.

Reversing the direction of travel does not avoid the arc, which is what the
mechanism predicts — the joint-space path between the same two
widely-separated poses is the same curve traversed either way. It clips
slightly *harder* in reverse (+4.36 vs +4.05 mm), so it is the tighter of the
two cases and the one any fix should be checked against. It is counted
against the `h1` target in the per-target table below, since that is the
square it places on.

**h1->h8 is the extreme member of a cluster, not an isolated case.** Peak
XY excursion beyond the bound across all 189 episodes (negative = margin
remaining):

```
  +145.13  c2->a1   FAIL   place       (D11, a different mechanism)
    +4.05  h1->h8   FAIL   traverse
   -16.50  h2->h8   ok     traverse
   -27.92  g1->h8   ok     traverse
   -37.02  a8->a1   ok     traverse
   -38.37  h3->h8   ok     traverse
```

187 of 189 never exceed the bound, and only one passing episode comes within
20 mm of it. But the near-bound cases form an ordered family — h1, h2, h3
-> h8 and g1, g2 -> h8 — all shaving the same edge during `traverse`. h1->h8
is the first to cross, not a singular outlier.

**Deterministic.** Identical `outcome` / `failed_phase` / `steps`
(`piece_off_board` / `traverse` / 76) across 5 seeds and across 3 repeat runs
at the same seed. This is expected rather than informative: with jitter=0 the
env reset is deterministic and the squares are forced, so the seed cannot
influence anything. There is no stochastic component here.

**Not a release mechanism.** Identical under condition (ii) (old instant
release) and (iii) (Candidate 1), and the episode terminates in `traverse`,
two waypoints before `release` is ever entered — the release code does not
execute on this episode at all.

**Correction to the original framing.** The first version of this entry
described both residual failures as "long diagonal traversals". That is
wrong on the measured geometry and would misdirect a fix: h1->h8 is a
straight 546 mm traverse along a single file with zero diagonal component,
and D11's c2->a1 is a 174 mm move of 2 files and 1 rank — one of the shorter
moves in the set. Neither is a long diagonal.

**Tested and rejected — reach extremity.** The earlier version of this entry
logged, as an explicitly untested hypothesis, that both residual failures
targeted a corner at ~99% of the UR5e's 850 mm rated reach (D2), and that the
3-target sweep could not separate "the target is a corner" from "the target
is not e5". The sweep was widened to five targets, adding the two remaining
corners `a8` and `h1`. The hypothesis does not survive it:

| target | planar reach from shoulder | failures / 63 starts | which |
|---|---|---|---|
| `a1` (near corner) | 0.3682 m | 1 | c2->a1 (D11) |
| `h1` (near corner) | 0.3682 m | 4 | a4->h1, g5->h1, g2->h1 (D11); h8->h1 (this entry) |
| `e5` (centre) | 0.5604 m | 0 | — |
| `a8` (far corner, ~99% reach) | 0.8387 m | **0** | — |
| `h8` (far corner, ~99% reach) | 0.8387 m | 1 | h1->h8 (this entry) |

6 failures across 315 episodes — 309 clean, 98.1%. Failures are counted
against the square the episode *places on*, regardless of which mechanism
they belong to. `a8` is h8's mirror-image corner at the identical 0.8387 m
planar reach and is **clean at 0/63** — if reach extremity drove this
failure, a8 would fail at h8's rate, and it does not fail at all. The two
*near* corners, 0.3682 m out and nowhere near the reach limit, carry 5 of the
6 failures (four of them D11's mechanism, per that entry's own target
association; the fifth is this entry's reverse pair h8->h1). Reach extremity
predicts neither failure and is dropped. The kinematic result above replaces
it: what distinguishes h1->h8 is the shape of
the joint-space arc to that target, not how far away the target is.

**Also unexamined:** `_piece_off_board()` is a pure XY/Z extent test with no
reference to grip state or height, so it fires on a securely gripped piece in
mid-transit. Whether that is the intended contract for the predicate is a
separate question from why the trajectory bulges, and neither has been
settled.

**The fix, and why it is off by default.** `_traverse_legs()` in
`scripts/scripted_expert.py` walks the pinch point along the straight line to
the traverse target in ~50 mm legs, re-solving IK at each and driving each for
3 macro-steps with no settle requirement, then lets the ordinary settle drive
finish on the final pose. Arc sag falls as ~1/N^2, so at 11 legs on the
longest traverse the residual bulge is three orders under the 78 mm margin —
structural, not tuned to the margin it has to clear.

*It works, on its own terms.* Measured peak excursion, against the same
predicate:

| pair | fix off | fix on |
|---|---|---|
| h1->h8 | **+4.05 mm** (fails) | **-74.65 mm** — success, 8.2 mm from goal |
| h8->h1 | **+4.36 mm** (fails) | **-75.82 mm** — success, 3.9 mm from goal |

*It is still disabled*, because it does not leave the rest of the board
alone. Three runs of the same 5-target / 315-episode sweep:

| variant | result | failures |
|---|---|---|
| fix off (baseline) | 309/315 (98.1%) | a4->h1, c2->a1, g2->h1, g5->h1, h1->h8, h8->h1 |
| fix on | **311/315 (98.7%)** | b2->h1, c3->h1, f2->a1, h2->a1 |
| fix on, final pose re-solved from the post-walk configuration | 308/315 (97.8%) | a8->e5, b8->e5, c2->a1, c3->h1, f7->e5, g3->e5, h3->a1 |

**No pair fails in all three runs**, and 13 distinct pairs appear as failures
across them. D10's own two pairs are the only ones that behave
deterministically: they fail with the fix off and pass with it on, in both
variants. Everything else is the D11 population being resampled. The best
aggregate of the three (98.7%) is therefore not evidence that the fix is
safe — it is one draw from a distribution whose spread is wider than the
difference between the variants. Four episodes that passed at baseline fail
with the fix on, and that is the number that governs. See D11's "Upstream
perturbation resamples the failure set".

Committed behind `TRAVERSE_TASK_SPACE = False`. With the flag off the branch
never runs: a flag-off sweep reproduces the baseline in all 315 episodes,
identical in outcome, failing phase, step count and final distance, so the
default path is unchanged by this commit. Turn it on when D11 is root-caused,
not before.

**A trap this mechanism sets for any future fix.** The measured excursion is
4.05 mm but the *planned* arc overshoots by 25-27 mm — physics currently
reduces the excursion by ~20 mm, because the arm never fully tracks the
commanded corner. Any change that makes the arm follow the commanded
joint-space path more faithfully — more joint-space waypoints, a tighter
settle tolerance, slower motion — therefore makes D10 **worse**, not better.
Only re-specifying the path in task space changes the arc itself.

**Also considered and not adopted as the fix:** relaxing `_piece_off_board()`
so it ignores a gripped, airborne piece. That removes the symptom while the
arm still swings outside the board footprint, which in a physical cell is a
collision-envelope question rather than a bookkeeping one. Whether the
predicate's contract should change is a real question, kept separate below.

**Forward-note for T2 — the fix assumes free space at `CLEAR_Z`.** A
straight line between the two transit poses is only safe because in T1 the
volume above the board is empty: one piece, no obstacles, both endpoints at
the same height. Note the direction of the old failure — the joint-space arc
bulged *outward*, away from the board, which meant it was incidentally clear
of anything standing *on* the board. The straight line is not. **T2's obstacle
task must re-verify this before reusing the traverse logic**: with other
pieces on the board a straight line at `CLEAR_Z` can pass directly over or
through them, and the fix would need to become a collision-aware path rather
than a linear interpolation. The assumption is recorded in the constants block
in `scripts/scripted_expert.py` as well, where a T2 implementer will meet it.

**Consequence:** not a blocker for D8/D9 — both are release-phase fixes and
this occurs upstream of release. Still open in effect, since the fix is
disabled: the two pairs continue to fail on the default path, and the cluster
above says the margin on the passing cases is thin enough that a change of
transit height or board placement could put more episodes over the bound.
D10 cannot be called resolved, and no "100% board coverage" claim can be made,
until the flag can be turned on — which is gated on D11.

---

## D11 — pad-load asymmetry during `place` squeezes the piece out of a still-closed gripper; 4 known instances, varying aftermath

**Date:** 2026-08-27 (logged as part of D10); split out 2026-09-10; scope
widened to 4 instances 2026-09-10 after the a8/h1 target sweep
**Status:** open, not root-caused. No fix attempted.

**Scope note:** this entry was first written around c2->a1 alone and framed
as "piece ejected". A later sweep adding targets `a8` and `h1` found three
more failures with the same measured signature, and showed that the ejection
is only the most violent of several aftermaths. The entry now covers all four
as one mechanism. The severity range is wide enough that the outcome label
and the failing waypoint differ between instances — see "Labels mark
detection, not cause" below.

**The four instances:**

| pair | grip lost at | peak pad force | aftermath | `outcome` | `failed_phase` |
|---|---|---|---|---|---|
| c2->a1 | step 117, `place` p22 | 318.6 N | ejected at 151.86 m/s to z=2.07 m | `piece_off_board` | `place` |
| a4->h1 | step 123, `place` p20 | 504.1 N | pops out, lands, slides off (+16.13 mm) | `piece_off_board` | `place` |
| g5->h1 | step 128, `place` p23 | 370.3 N | drifts free, trips bound later (+24.44 mm) | `piece_off_board` | `release` |
| g2->h1 | step 114, `place` p21 | 229.8 N | stays on the board, ends 290.9 mm from goal | `sequence_finished_not_at_goal` | none |

**Shared signature.** In all four, with the gripper still commanded closed
(aperture ~0.63 throughout, no release command issued):

1. Pad normal load becomes strongly asymmetric over roughly 5-10 macro-steps
   while the piece stays rotationally quiet.
2. The piece's angular speed jumps suddenly — 32.59, 39.11, 27.64 and 35.22
   rad/s respectively — **exactly one macro-step before** peak total pad
   force.
3. Total pad force peaks (229.8-504.1 N against a ~44 N steady-grip
   baseline), contact count collapses, and the piece leaves the pads.

**Labels mark detection, not cause.** g5->h1 is recorded as failing in
`release`, but its last pad contact is step 128 and the `release` waypoint
does not begin until step 130 — the piece is already free two steps before
release starts. It fails identically under both release conditions
(`--release old` and `--release new`, step 141 both), which is expected once
the causal event is located in `place`. g2->h1 never sets `piece_off_board`
at all and is only visible as a distance failure. Classifying these by
`failed_phase` or `outcome` would have split one mechanism into three.

**Tested and not supported: pre-existing rotation (a D8 carry-over).** If
this were D8's rotationally-underconstrained head-only grasp finding a second
entry point during `place`, the piece should already be rotating on the way
in. It is not. Peak angular speed over the whole `place` phase before the
rotation onset is 0.44, 0.67, 0.94 and 0.70 rad/s for the four failures —
squarely inside the range measured on successful places at the same targets
(0.26 to 4.47 rad/s across eight controls). The rotation is a *consequence*
of the load event, not a precondition. What is *not* excluded: the rotation
onset magnitudes (27.6-39.1 rad/s) overlap D8's observed release spin
(12.8-30.3 rad/s), and the grasp geometry is the same head-only grasp D7
committed to, so a shared underlying susceptibility remains plausible — it
simply is not evidenced by pre-spike rotation.

**Tested and not supported: asymmetry as the discriminator.** Peak left/right
imbalance during `place` does not separate failures from successes. Failures
span 136.0-223.2 N; successful controls at the same targets reach comparable
or higher values (f6->h1 199.9 N, e7->h1 183.8 N, d4->h1 168.8 N, c3->a1
149.0 N), and c3->a1 even shows a rotation onset at the same `phase_step` as
the failures and still succeeds. Peak *total* force does not separate them
either (failures 184.0-240.7 N, controls 201.0-239.7 N). So the asymmetric
loading is a normal feature of placing at these squares, not a fault
signature, and whatever tips a marginal case into failure is not captured by
either measure. This echoes D8's own "chaotic in outcome" characterization.

**Target association.** All four instances place at `a1` or `h1` — the two
*near* corners, 0.3682 m planar from the shoulder. Zero instances at `e5`
(0.5604 m) or at `a8`/`h8` (0.8387 m, the ~99%-of-rated-reach corners).
Asymmetry does track corner placement — the `e5` controls peak at 20.0 and
26.8 N imbalance versus 123-223 N at corner targets — but, per the paragraph
above, it does not predict failure among corner placements. Five targets is a
thin basis and no cause is proposed.

**Upstream perturbation resamples the failure set (new lead, 2026-09-10, from
D10's fix work — not chased).** D10's fix changes only `traverse`, one
waypoint upstream of `place`. Running the same 5-target / 315-episode sweep
with it off, on, and on in a variant that also re-solves the final traverse
pose gives 6, 4 and 7 failures — and **no pair fails in all three runs**. 13
distinct pairs appear as failures across them. D10's own two pairs are the
only ones that move deterministically; everything else here moves.

What makes this more than noise is that the piece's own state entering
`place` is *unchanged*. On the four instances above, with the fix off vs on:

| pair | entry lin (m/s) | entry ang (rad/s) | peak force in `place` (N) | peak ang in `place` (rad/s) | outcome |
|---|---|---|---|---|---|
| c2->a1 | 0.391 / 0.390 | 0.045 / 0.046 | 625.2 / 246.2 | 71.4 / 71.5 | fails / fails |
| a4->h1 | 0.395 / 0.392 | 0.064 / 0.065 | 506.3 / 248.1 | 104.7 / **0.67** | fails / **succeeds** |
| g5->h1 | 0.396 / 0.393 | 0.074 / 0.066 | 556.5 / 254.8 | 42.4 / **0.60** | fails / **succeeds** |
| g2->h1 | 0.392 / 0.392 | 0.072 / 0.071 | 366.3 / 308.6 | 113.3 / 54.9 | fails / **succeeds** |

Entry linear and angular speed agree to three decimals in every row, while
peak pad force halves and the outcome flips. So whatever the fix changed, it
is not what `place` inherits in the piece's velocity — the candidate is the
**arm's joint configuration at the start of `place`**, which sets the
joint-space path `place` takes down to the board. That is D10's own mechanism
applied to a different waypoint, and it is the first variable proposed here
that is about the *arm* rather than the piece and the pads. Every measure
tested in this entry so far — pre-existing rotation, asymmetry magnitude,
peak total force — is a property of the piece or the contact, which may be
why none of them discriminated.

Two cautions before anyone chases it. First, the "no re-solve" variant holds
the final commanded traverse pose *identical* to baseline and still resamples
the failure set, so configuration at the waypoint's end cannot be the whole
story — the route taken to it matters too. Second, and more important, a
resampled failure set at a roughly constant rate is the signature of
sensitive dependence, not of a discriminator waiting to be found: it may be
that no upstream variable predicts individual failures and only the *rate* is
meaningful. That would make "which squares fail" the wrong question, and it
is consistent with this entry's existing "chaotic in outcome"
characterization and with D8's. Either way, an aggregate improvement measured
on one draw (D10's fix scoring 98.7% against a 98.1% baseline) is not
evidence of a real gain — the spread across variants is wider than the
difference between them.

**Reclassified 2026-09-11: c2->a1 belongs to D14, not here.** D14's guard was
sized against measured piece kinetic energy, and measuring these four
instances separates them cleanly:

| instance | peak piece KE | reads as |
|---|---|---|
| g2->h1 | 0.0665 J | D11 |
| g5->h1 | 0.2143 J | D11 |
| a4->h1 | 0.2247 J | D11 |
| **c2->a1** | **224.90 J** | **D14** |

c2->a1 is three orders of magnitude outside the range of the other three and
within a factor of three of D14's own f4->h1 event (563 J). Its 151.86 m/s
exit was always the outlier of this entry -- the original text called it
"characteristic of a contact-solver blow-up rather than of the piece being
pushed or dropped off the board", which is exactly what D14 later established
independently, with the commanded side measured and ruled out. It is
therefore counted under D14 from here on, and under D14's guard it now
reports `terminated_reason = "solver_instability"` rather than
`piece_off_board`.

The historical record above is left as written rather than rewritten: the
measurements in this entry were taken before D14 existed and are correct as
measurements. What changes is which mechanism they are attributed to. The
"four instances" framing of this entry now covers three.

**SUBSTANTIALLY RESOLVED (2026-09-11), as a side effect of D14.** Widening
the piece's contact time constant from `solref = 0.004` (2x timestep) to
`0.008` clears every instance in this entry. On the five-target 315-episode
sweep, T1 goes from 309/315 to **313/315**, and the four failures that
disappear are precisely a4->h1, c2->a1, g2->h1 and g5->h1 -- this entry's
whole population -- with **zero newly broken episodes**. The only remaining
failures on the board are D10's two structural pairs.

So the mechanism this entry describes -- pad load going asymmetric and the
piece being squeezed out of a still-closed gripper -- appears to have been a
contact-stiffness artefact rather than a property of the head-only grasp D7
committed to. That also explains why no measured variable ever discriminated
failures from successes here: the discriminating variable was not in the
grasp at all.

Two reasons this says "substantially" and not "resolved":

1. Per this entry's own resampling finding, one draw is one draw. What makes
   this more than a lucky resample is the *shape* of the result -- every
   previous perturbation moved the failure set to different pairs at a
   similar count (6, 7, 4, 6, 9), whereas this shrinks it to exactly the two
   deterministic kinematic pairs and adds nothing. But it has been measured
   at one alternative `solref` value, not swept.
2. Nothing here was root-caused in the sense D14 was. No trace shows *why*
   the marginal loading became unstable at 2x timestep; the evidence is that
   it stops happening at 4x.

**Consequence:** no longer a blocker for anything, and no longer the blocker
for D10: D10's fix is implemented, validated on its own terms, and
held disabled behind `TRAVERSE_TASK_SPACE` because enabling it trades four
previously-passing episodes for the two it fixes, and D11 is not understood
well enough to say whether that trade is real or an artifact of resampling.
Does need root-causing before any "100% board coverage" claim.

---

## D12 — T2's observation gains the obstacle positions, added at the wrapper, not in `RoboChessEnv`

**Date:** 2026-09-10 (T2 design session)
**Status:** decided and implemented.

**The problem.** `RoboChessEnv._get_obs()` builds its vector by explicit index
— arm qpos/qvel, gripper qpos/qvel, then the tracked piece's 7 qpos — so
adding obstacle bodies to a scene does **not** change the observation. T2's
obstacles would therefore be invisible to a policy, and T2 would be a task
whose central difficulty the agent cannot perceive: it could only learn a
cautious average policy, never obstacle avoidance. Measured incidentally: the
observation stayed 21-dimensional with four obstacle bodies present, and T1's
`test_reset_returns_correctly_shaped_obs` passed unchanged.

**Decision.** T2's observation appends the (x, y) of every pool slot, taking
the vector from 21 to 29 dimensions.

**Where it is applied: on the wrapper, not the env.** `T2ObstacleTask`
declares its own `observation_space` and extends the dict returned by
`reset`/`step`. `RoboChessEnv._get_obs()` and its `observation_space` are
untouched, as is `t1_pick_and_place.py`. T1's 48 existing tests pass
unmodified, which is the check this rests on rather than inspection. The
alternative — widening the base env — would have changed T1's observation
contract to serve a task T1 does not have, and would have put D5's
locked state-vector HER configuration in scope for a T2 change.

**Constant size regardless of obstacle count.** All four slots always report,
and an unused slot reports its fixed parked position (`PARKED_XY`, on the
floor at (1.20 + 0.20i, -1.20)). A vector whose width varied with
`n_obstacles` would make the observation space a function of an episode
parameter, which P4's factorial cannot carry. The parked coordinates are a
documented constant for this reason, not an arbitrary "somewhere off to the
side".

**Consequence for the reward arms, which must be named in the writeup.**
`ShapedPickAndPlaceReward` and `SparsePickAndPlaceReward` are reused
unchanged, which is the reusability goal T1's module docstring states, and
the sparse arm picks up obstacles for free because T2 amends
`info["is_success"]` with its fifth success term. The shaped arm does not.
Its reward is a pure function of two distances — piece-to-goal and
gripper-to-piece — and it never reads `is_success`, so **it receives no
per-step signal about obstacles at all and can only avoid them by accident.**

That is a real asymmetry between the two arms of the factorial, not a
detail: any shaped-vs-sparse comparison on T2 is partly a comparison between
a reward that can see the task's obstacles and one that structurally cannot.
**This must be stated explicitly when T2's results are analysed**, not left
for a reader to infer from the reward definitions. A third
`ObstaclePenaltyReward` wrapper was considered and rejected for now: it would
add a reward formulation the study design does not have.

A related asymmetry already documented for T1 extends to obstacles:
`compute_reward`, which HER uses to score relabelled goals, sees only
`(achieved_goal, desired_goal)` and so cannot score obstacle disturbance —
the same limitation that already applies to `released` and `settled`.

---

## D13 — T2 gets its own generated scene; the obstacle pool must not go in the shared `scene.xml`

**Date:** 2026-09-10 (T2 implementation session)
**Status:** decided and implemented.

**Finding.** Adding T2's four free-jointed obstacle bodies to `assets/scene.xml`
changes T1's behaviour, even though the obstacles are parked 1.2 m away on the
floor and never touched by any episode:

| | five-target sweep (315 episodes) |
|---|---|
| committed `scene.xml` | **309/315 (98.1%)** |
| `scene.xml` + 4 parked obstacle bodies | **306/315 (97.1%)** |

**149 of 315 episodes differ numerically.** Seven cross the success line:
`g2->h1` and `g5->h1` start passing; `b2->a1`, `b2->h1`, `c3->h1`, `f4->h1`
and `f8->e5` start failing, one of them flinging the piece 20 m. Extra DOFs
change the constraint solve, and per D11 the marginal placement population is
resampled by any such perturbation. D10's two pairs are bit-identical across
both scenes (76 and 143 steps, same distances), as expected — that mechanism
is structural, not marginal.

**Why this mattered enough to change the design.** File-level isolation was
not sufficient. `robochess_env.py`, `t1_pick_and_place.py` and `tests/` were
untouched and all 48 tests passed — and T1's validated 309/315 still stopped
reproducing. Every sweep figure recorded in D8, D9, D10 and D11 refers to the
model as it was; sharing the scene would have silently made all of them refer
to a different model.

**Decision.** T2 loads `assets/scene_t2.xml`, via `RoboChessEnv`'s existing
`scene_path` argument. `scene.xml` is unchanged, byte for byte.

**Generated, not hand-maintained.** `scripts/build_t2_scene.py` produces
`scene_t2.xml` from `scene.xml`; the generated file is committed, and a unit
test in the normal suite asserts that regenerating reproduces it byte for
byte, so the two cannot drift when board or piece geometry changes. The
generator is the only thing that writes the file, and it says so in a header
comment.

**Consequence for T2's numbers.** T2's control is **not** T1's 309/315.
The pool bodies perturb the solve whether or not obstacles are placed, so the
only honest control for T2 is T2's own scene with `n_obstacles=0`, measured
on the same five targets. Any T2-vs-T1 comparison that skips that control is
comparing two different models.

---

## D14 — solver-side contact energy injection: the physics, not the controller, produces a ~600 J event

**Date:** 2026-09-10 (T2 validation session)
**Status:** **RESOLVED** (2026-09-11): root-caused to the piece's contact
time constant sitting at MuJoCo's documented minimum, fixed by widening it,
and additionally guarded in `RoboChessEnv` so the failure class is caught
whatever its cause. No longer a P3 blocker. Deliberately NOT filed as a D11
instance — see "Why this is not D11" below.

**Finding.** On `f4->h1`, the tracked piece acquires **682 J of kinetic
energy** and 4224.69 rad/s of angular speed while the arm is essentially
stationary and holding a converged joint target. The episode ends
`piece_off_board` at step 117 with the piece **20.3 m** from the goal.

```
step pstep  lin m/s  ang rad/s      KE (J)  arm qvel  qfrc_con  ncon
 115    19     0.29      37.51      0.0038     0.064      11.0    26
 116    20   234.62    3541.22    563.5395     0.071    1411.5    24
 117    21   257.02    4224.69    682.2587     0.069       0.2    12
```

**The controller is not the source, and this was measured rather than
assumed.** At the blow-up step:

| quantity | value | normal? |
|---|---|---|
| commanded-vs-actual joint gap | 0.008 rad (0.5 deg) | arm is *at* its target |
| `ctrl` | -2.7787 ... -1.2079 | range is +/-6.283; nowhere near a limit |
| `ctrl` ever out of range, whole episode | never | — |
| max normalized action | 1.000 | inside the [-1, 1] action space |
| max actuator torque | 25.7 Nm | forcerange is 150/28 Nm |
| max arm joint velocity | 0.069 rad/s | the arm is nearly still |

The piece masses 0.0195 kg and gains **563.5 J inside a single 0.04 s
macro-step (~14 kW)**, while the actuators, moving at 0.07 rad/s under
<= 26 Nm, can deliver on the order of **1.75 W**. That is four orders of
magnitude of energy the controller cannot have supplied, alongside a
`qfrc_constraint` spike to 1411.5 and pad contacts collapsing to zero. The
energy is injected by the constraint solve.

**Deterministic, and not obstacle-related.** Identical across three repeat
runs at jitter=0, seed=0, and identical at 0, 2 and 4 obstacles — placed or
parked makes no difference.

**Triggered by the presence of extra bodies, not by their behaviour.** The
same pair is a clean success on T1's scene:

| model | result |
|---|---|
| `scene.xml` (baseline, and 3 traverse-fix variants) | success, 135 steps, 1.3 mm |
| `scene_t2.xml` (obstacle pool present, parked or placed) | 20256.5 mm |

This is the D11 resampling sensitivity again — extra DOFs change the
constraint solve — but the *magnitude* here is a different thing entirely.

**Why this is not D11.** D11's four instances run 229.8-504.1 N of pad force
with rotation onsets of 27.6-39.1 rad/s, and are consistent with a piece being
squeezed out of a gripper. This is 3541-4225 rad/s and 682 J with the arm
holding still: two orders of magnitude past D11's measured range, with no
commanded motion to explain it. D11's own entry does already read
"characteristic of a contact-solver blow-up", so the two may eventually merge
— but merging them now would lose the distinction that matters, which is that
this one is provably independent of what the controller does.

**Consequence for P3, which is why this blocks rather than annoys.** A policy
under PPO/SAC drives the same position actuators through the same
`_scale_action` clip into the same model. Nothing about the scripted expert
was protective: its commands were unremarkable and in range. A learning
policy will visit far more pad-piece contact configurations than a tuned
waypoint sequence does, and a single such event injects hundreds of joules
into a transition the agent then trains on. Any pilot-matrix cell that hits
one is learning from corrupted data, and neither the reward nor the success
criterion currently notices.

**RESOLVED (2026-09-11) by widening the piece's contact time constant, plus
a guard that stands regardless.**

The observation logged with this entry -- that the piece geoms carried
`solref = [0.004, 1]` against `timestep = 0.002`, a contact time constant of
exactly 2x the timestep, MuJoCo's documented minimum -- turned out to be the
cause. Tested in isolation on f4->h1, three runs each:

| piece `solref` | outcome | peak angular speed | peak KE |
|---|---|---|---|
| `0.004` (2x timestep, as committed) | `piece_off_board`, 20256.5 mm | 4224.7 rad/s | **682.26 J** |
| `0.008` (4x timestep) | **success**, 5.0 mm | 2.7 rad/s | 0.0184 J |
| `0.02` (the board slab's value) | success, 6.7 mm | 1.4 rad/s | 0.0177 J |

**`solref = 0.008` is adopted; `0.02` is not.** The softer value fixes
f4->h1 too but is catastrophic elsewhere -- a full 315-episode T1 sweep at
`0.02` scores 283/315 (89.8%) and breaks 29 previously-passing episodes,
almost all long traverses to h1. This is exactly why the change was measured
across the whole board before adoption rather than validated on the pair it
was designed to fix.

**The guard (`_solver_instability`) is kept even though the cause is fixed.**
Nothing establishes that this was the only reachable unstable configuration,
and P3 needs a policy that meets one to fail loudly rather than train on the
transition. Measured cost of keeping it: a full T1 sweep with the guard active
and the old `solref` returns the identical 309/315 and the identical failure
set -- zero false positives in 315 episodes. Measured margins: ordinary
successful episodes peak at 0.012-0.032 J, ordinary D11 instances at
0.067-0.225 J, the threshold is 1 J, and the two real events sit at 224.90 J
and 563.54 J.

Two details of the guard worth knowing:

* It **truncates rather than terminates** -- a solver artefact is not an MDP
  terminal state, and labelling it terminal would teach a value function that
  this configuration is a genuine end-state of the task.
* On trigger it returns the **last known-good observation**, not the current
  one. SB3 bootstraps the value function off the final observation when an
  episode is truncated (unlike termination, where it does not), so returning
  the spike state would feed it straight to the value estimate -- the one
  path by which the corrupted state would still reach the learner despite the
  episode ending. Measured on f4->h1: the returned `achieved_goal` is
  `[0.2707, 0.2476, 0.7597]`, the piece resting where it was before the
  spike, instead of `[-5.0, 2.3562, 5.0]` -- the corrupt state clipped to the
  observation space's own bounds.

---

## D15 — T2's obstacle exclusion is a kept precaution, not a measured requirement; and T2 as configured tests clutter, not avoidance

**Date:** 2026-09-11 (T2 validation session)
**Status:** decided, and recorded chiefly so the second half is not forgotten
when T2's results are written up.

**What the exclusion is.** `T2ObstacleTask` draws obstacle squares at random,
excluding the start square, the target square, and the *file-adjacent*
neighbours of both (`file_adjacent_squares`). The stated reason was that the
2F-85's open fingers overhang the file-adjacent square during `descend`.

**That reason was an overestimate, and the correction belongs on the record.**
It came from sweeping every moving geom against a standing piece's envelope
with each geom treated as its MuJoCo **bounding sphere**, which put
`ur_gripper_left_pad1` ~4 mm inside the file-adjacent square's envelope on
every start square tested (rank-adjacent squares ~14 mm clear). Bounding
spheres are larger than the pad boxes they enclose. Measured against real
contacts instead:

* An obstacle standing on a file-adjacent square is **never touched**. Over a
  full episode its only contact, every step, is `obstacle_1_base` against
  `board_slab`.
* Forcing an obstacle onto a file-adjacent square (d4, f4, g1) or onto the
  transit line (e5) produces no displacement at all: 0.00 mm, term never
  fires, episode succeeds.
* The measurement path is not silently broken -- forcing an obstacle onto the
  *target* square does fire it, at 15.29 mm (e4->e6) and 174.21 mm (a1->h8).

**Decision: keep the exclusion anyway.** It costs 2-4 squares out of 64, the
margin is small either way, and a bystander adjacent to the square being
grasped is the placement most likely to become unsafe if the grasp geometry,
the approach, or `CLEAR_Z` changes. What must not happen is the exclusion
being cited as a measured collision requirement, which is what the original
docstring did; that wording is corrected.

**The part that matters for the writeup.** Across both five-target
315-episode T2 sweeps -- `n_obstacles=0` and `n_obstacles=2` -- the obstacle
term fired **0/315 in both**, and the maximum obstacle displacement anywhere
was 0.000 mm. The scripted expert makes no attempt whatsoever to avoid
obstacles, and still never disturbs one.

**Correction (F6, referee-repairs).** Those original figures were read from
the terminal step's snapshot, and the disturbance check had no episode memory:
an obstacle knocked and then put back read "undisturbed" again. So the 0/315
and 0.000 mm were a measurement of the end state, not proof that no obstacle
was ever disturbed mid-episode. F6 added a per-episode latch
(`obstacles_ever_disturbed`, `max_obstacle_peak_displacement_mm`) and
`scripts/fullboard_validation_t2.py` now reports it. Re-measured with the
latch on the same five-target sweep, `n_obstacles=2`, 315 episodes: **latch
fired 0/315, maximum peak displacement 0.000003 mm** (303/315 successes; the
303, not 313, is the post-F1 baseline, see the D24 note on the d1/e1 starts).
So the corrected, latch-based figure is also 0/315. The `n_obstacles=0`
sweep places no obstacles, so its 0/315 is trivially true and says nothing
about disturbance. Limits that still apply: the latch is sampled once per
env step (a knock-and-return inside one 20-substep macro-step is not seen),
it covers the scripted expert's single fixed trajectory family only, and the
10 `d1`/`e1` episodes end at step 37 on `bad_contact`, so they say little
about the rest of their trajectories.

So, stated plainly: **T2 as currently configured measures task completion
with static clutter present on the board. It does not measure active
obstacle avoidance under pressure.** The transit corridor clears a standing
piece, the exclusion removes the only squares the gripper comes near, and
nothing in the task forces the arm to route around anything. That is a
legitimate first version of an obstacle task and it exercises the machinery
-- placement, the fifth success term, the observation extension -- end to
end. It is not evidence that a policy can avoid obstacles, and no T2 result
should be reported as if it were.

What would change that, if a later phase wants the harder task: allow
obstacles on the excluded squares, raise the obstacle count toward the full
pool of 4, place obstacles deliberately on the straight line between start
and target, or lower `CLEAR_Z` so transit no longer clears a standing piece
by ~9-13 mm. Each is a deliberate difficulty increase, none is a bug fix, and
none has been attempted.


## D16 — action space: absolute joint targets are the root cause of the pilot's 0% success; delta mode added behind a flag, absolute kept as the default

`_scale_action` mapped each arm action linearly across the actuator's full
ctrlrange as an ABSOLUTE joint target: +-1 became +-2 pi rad on five joints and
+-pi on the elbow, with no reference to the previous target or to where the
joint actually was. Action 0 is therefore not "hold still" -- it commands every
arm joint to 0 rad, far from the home pose. This was the action space all 16
pilot cells trained on.

Measured consequences under absolute mode (results/pilot_diagnosis):

* both trained policies sat near +-1 on most steps;
* median command-vs-actual joint error up to 295 deg;
* the big-joint actuators were at their force limit on 88-99% of steps, with the
  joints nowhere near their own limits -- purely the force cap;
* the gripper was never brought over a board square, and the arm was never
  quasi-static (0 of 600 steps).

Everything the pilot reported is downstream of this: no approach, never
quasi-static, "hold still" as the only learnable way to avoid the terminal
penalty, and D14-guard triggers from force-saturated links striking a resting
piece. It also does not match the proposal's section 4.2.2, which specifies the
joint mode as per-joint position DELTAS.

**Fix, shipped behind a flag (commit `93b4ee6`).** `RoboChessEnv(action_mode=)`
takes "absolute" (default) or "delta". In delta mode each arm action scales to a
per-step CHANGE in the joint target, capped at `max_joint_delta`, applied to the
previous target and clamped to the joint's range; the gripper stays absolute in
both modes so a policy can open or close in one step.

**Why the cap is 0.1 rad/step.** Calibrated from the scripted expert's ACTUAL
joint motion over 23 successful episodes (p95 0.04-0.08 rad/step on every joint;
p99 0.11-0.14 on pan/lift/wrist_1/wrist_3, elbow 0.19, max 0.31). The expert's
COMMANDED targets could not be used: it is a waypoint-jump controller whose
target is unchanged on 95.6% of steps and then jumps up to 1.85 rad, which would
have reproduced the original problem. 0.1 also sits just past the actuator
saturation threshold (forcerange/kp = 0.075 rad on the big joints, 0.056 on the
wrists).

**Absolute stays the default** so every pilot run remains byte-for-byte
reproducible; the flag-off path was verified byte-identical on fixed-seed T1 and
T2 traces. Same pattern as D13.

**The D7 follow-on question is answered by the same diagnosis.** The RL action
does reach the same uncompensated PD servos the scripted expert needed D7's
compensation for (kp 2000/500, `body_gravcomp` 0 everywhere, action passing
through the wrappers unmodified -- confirmed at runtime, not by reading code).
But D7's ~0.6-1.5 deg droop is two to three orders of magnitude smaller than the
error these policies operate at, so gravity droop is not the operative effect
for trained policies; saturated extreme-range commands are.


## D17 — SAC critic divergence under the raw reward scale — OPEN

One pilot cell, `t2_sac_shaped_seed1`, diverged: critic loss rose from 7.5 to a
peak of 1.0 x 10^8, the entropy coefficient reached 3.03, and behaviour degraded
in lockstep. Every other SAC cell peaked below 660. **That cell's results are
invalid** -- its later checkpoints come from a diverged critic, and the 100k
checkpoint is the last clean one.

Leading hypothesis, UNTESTED: PPO is wrapped in `VecNormalize` and SAC
deliberately is not, so SAC's critic fits raw targets where most transitions sit
near -0.5 and rare ones at the terminal penalty (-257 at the 100-step episode
length those cells used). Heavy-tailed bootstrap targets are a known route to
critic blow-up.

**Note for the writeup, to avoid a wrong argument:** do NOT claim that a positive
Q proves divergence "because all rewards are negative". Maximum-entropy SAC
bootstraps the entropy bonus, so a positive Q is possible in principle. The
critic-loss gap (1.0 x 10^8 against every other cell's <= 660) is the evidence,
and it stands on its own.

Still to decide after the re-pilot: normalise SAC's rewards, or reduce the
penalty magnitude. Neither has been done.


## D18 — `grasp_success` counted grasps that never happened

The original definition was "piece lifted more than `rest_tolerance` above the
board AND gripper aperture at/above `gripper_open_threshold`" -- lifted AND
closed. A piece knocked onto its side satisfies "lifted" all by itself, because
its origin rises ~18 mm when it topples, and a gripper closed on nothing
satisfies "closed". In the pilot diagnosis **all 50 flagged episodes were false
positives of exactly that kind, with 0 real grasps; in 49 of them the pinch site
never came within 5 cm of the piece** (18-68 cm in the traced cases).

**Fix, in two parts.**

1. (`93b4ee6`) Require BOTH finger pad bodies to be in contact with the piece,
   read from MuJoCo's contact list via `model.geom_bodyid`. Validated in both
   directions: the scripted expert's grasps still count exactly as before
   (50/50, 55/55, 28/28 steps, including D10's h8->h1 genuine-grasp-then-carry-
   off-board case), and all 19 previously flagged policy episodes now count 0.
2. (`97ba93e`) Require the piece NOT to be in contact with the board. The pad
   term alone left a gap: a toppled piece lying on the board with both pads
   closed on it is lifted (origin ~18 mm up), closed, and gripped. Verified by
   building that state with the expert's own IK and drive at a sweep of pinch
   heights: at +11/+14 mm the piece lies at 90-100 deg with both pads and the
   board touching it every step, and the old definition fires on 9-19 steps
   while the new one fires on 0. Real carries are unaffected (50/55/28/41
   flagged steps, identical to the pad-only definition, 0 disagreements).

**Any grasp count in this project's history measured before `93b4ee6` is
unreliable.** Remaining known gap: when the closing pads squeeze a lying piece
UP off the board, it can flag for 2-6 transient steps while genuinely held clear
of the board by the pads. Not closed.


## D19 — delta-target windup, and the clamp that removed it (saturation only partly)

With delta mode on, the commanded target accumulates deltas without bound
relative to the actual joint, so it can run far ahead of a joint that cannot
follow. Measured in D16's 100k sanity run: at 100k steps the shoulder_lift
target sat a median **100 deg** ahead of the actual joint, and force-limit
saturation ROSE over training (20k: 17/17/1/0/0/0% -> 100k: 53/66/36/36/28/0%)
rather than staying low. Classic integrator windup. Absolute mode at the same
budget was worse (79/87/89/65/58/3%), so delta mode was already an improvement;
windup is why it was not the full improvement.

**Fix (commit `cd39ec2`).** Each step the target is clamped to within
`2 x forcerange/kp` of the ACTUAL joint position, per joint, before the existing
joint-range clamp. The margin is derived from the live model, not hardcoded:
0.150 rad on shoulder_pan/lift/elbow, 0.112 rad on the wrists. The gripper stays
absolute and unclamped, and absolute mode is byte-identical (fixed-seed T1/T2
traces of obs, reward, done flags, ctrl and qpos compared before and after).
`forcerange/kp` is the position error at which the PD servo saturates; the
multiple of 2 was chosen, not derived.

**Result: windup removed, saturation only partly.** Re-running the same 100k
cell with the clamp on (`results/d19_clamp/`):

| | unclamped @100k | clamped @100k |
|---|---|---|
| median \|target - actual\| | up to 100 deg (lift) | 5-7 deg, every joint, every checkpoint |
| saturation, pan/lift/elbow/w1/w2/w3 | 53/66/36/36/28/0% | 17/23/29/20/8/0% |
| trend over training | rising | flat |
| median closest pinch-to-piece | 0.360 m | 0.266 m |

Saturation no longer rises, but 20-42% on pan/lift/elbow/w1 remains at some
checkpoints, and at 20k the clamped run was HIGHER than the unclamped one on
elbow (0.42 vs 0.01) and wrist_1 (0.38 vs 0.00). That follows from the margin
itself: the actuator saturates at 1x forcerange/kp, and a 2x margin still admits
errors between the two. A tighter margin has not been tried.

**Separately measured (`results/cartesian_feasibility/`): the clamp limits
achievable joint speed by 2.8-5.0x** -- 0.45-0.72 rad/s clamped against 1.9-2.3
rad/s pre-clamp, the latter close to the 2.5 rad/s the 0.1 rad cap allows per
40 ms step. This corrects an earlier claim in that file that the servo, not the
action space, set the plateau. Removing the clamp does not improve the
expert-replay success count (6/10 against the clamped 7/10), so it is not
obviously the thing holding the task back.

**D19 is therefore partly resolved:** the runaway target is gone, saturation is
capped but not low.


## D20 — the shaped reward has no grasp term, and its 3D piece-to-goal term charges for lifting; staged terms added behind a default-off flag, and they did not change behaviour at 100k

Note on numbering: D16-D19 are recorded in the roadmap document's decisions
table, not here (see D7's renumbering note for how that drift started). This
entry uses the roadmap table's numbering, which is canonical.

### The two defects, stated as the reward arithmetic

`ShapedPickAndPlaceReward` pays

    r = -(piece_to_goal_3D) - 0.3 * (gripper_to_piece) - failure_penalty[terminating failure]

Two consequences follow directly from that formula and neither was deliberate:

1. **No term rewards a grasp.** Closing the gripper changes neither distance,
   so nothing in the gradient tells a policy to close it. The only thing the
   reward pays for is moving the pinch site toward the piece and the piece
   toward the goal -- and the piece does not move unless it is already held.
2. **Lifting the piece is charged for.** The piece-to-goal term is 3D, so
   raising the piece off the board increases it. Measured on a scripted-expert
   d1->e5 episode: while the piece is carried, 3D-minus-XY distance is a median
   0.027 m and a maximum 0.061 m per step, 1.40 in total over that episode's 46
   carrying steps. Holding the piece up is strictly worse than leaving it down.

Together these make hovering near the piece a local optimum, and that is
exactly what the clamped delta 100k run did: approach improved (median closest
pinch-to-piece 0.427 -> 0.266 m across checkpoints) while real grasps stayed at
0 and episode return stayed flat (-53.31 +- 41.48 over 1011 episodes, per-step
-0.5373, quarter means -0.59/-0.51/-0.50/-0.55).

### The staged variant, and why those bonus values

Behind `shaped_staged` (config; default **False**, and flag-off behaviour is
byte-identical -- verified with fixed-seed 600-step traces of reward, done
flags and both distances through the real training stack, absolute and delta):

    r = -d_xy(piece, goal) - 0.3 * (gripper_to_piece)
        + grasp_bonus  if both finger pads contact the piece
        + lift_bonus   if, additionally, the piece is clear of the board
        - failure_penalty[terminating failure]     (unchanged)

The contact tests are `_piece_contacts()`, already written for the
grasp_success board-contact work; no new contact logic exists for this.

`grasp_bonus = 1.0`: sized against the distance terms' measured scale. The
worst non-terminal step on record is -1.1378 and the worst seen in a 3000-step
random-policy delta run is -0.8354 (median -0.4842). At 1.0 a grasped step
scores above every ungrasped step ever observed, so "grasping beats hovering"
holds regardless of where on the board the two happen -- which is the ordering
the unstaged reward lacks entirely. Below ~0.84 that ordering breaks somewhere
in the workspace.

`lift_bonus = 0.5`: sized against step-to-step noise rather than the distance
range. Consecutive-step reward changes in that same run are 0.0017 median and
0.0073 at p99, so 0.5 is ~70x the p99 fluctuation and cannot be lost in it. It
is also ~8x the largest per-step lifting cost the old 3D term charged (0.061).

**Known risk of the per-step form, not yet triggered:** success TERMINATES the
episode, so a grasped, lifted piece held over the goal pays roughly +1.5/step
for doing nothing, up to ~+450 over a 300-step episode, which beats placing it
and ending. A one-time bonus on first grasp and first lift would not have this
property. The per-step form is what was specified; this is recorded so the
first run that actually grasps is checked for it.

### Measured effect at 100k: none

100k steps, T1/PPO/state/seed 0, delta, `max_episode_steps` 300, against the
unstaged clamped run. 50 deterministic episodes per checkpoint, eval seed
10000, same script for both (`results/staged_reward/eval_staged.py`, which
separates "both pads on the piece" from "pads AND clear of the board", a
distinction `grasp_success` cannot make since it already requires both).

| ckpt | run | median closest (m) | closest | eps <10cm | pads | lift | grasp | success |
|---|---|---|---|---|---|---|---|---|
| 20k | unstaged | 0.427 | 0.189 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 20k | staged | 0.451 | 0.296 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 40k | unstaged | 0.421 | 0.204 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 40k | staged | 0.410 | 0.292 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 60k | unstaged | 0.318 | 0.132 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 60k | staged | 0.334 | 0.211 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 80k | unstaged | 0.258 | 0.041 | 4/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 80k | staged | 0.310 | 0.160 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 100k | unstaged | 0.266 | 0.042 | 3/50 | 0/50 | 0/50 | 0/50 | 0.00 |
| 100k | staged | 0.262 | 0.101 | 0/50 | 0/50 | 0/50 | 0/50 | 0.00 |

Across all 250 staged eval episodes the gripper never touched the piece once.
The mechanism is not subtle: both bonuses are step functions that pay only
once both pads are already on the piece, so a policy sitting 26-45 cm away
collects neither and sees no gradient from them at all. A sparse term added to
a dense one is still sparse. The XY switch is equally inert until something is
held.

This does NOT refute the diagnosis above -- the two defects are properties of
the formula and remain real. What it establishes is narrower and worth stating
plainly: **adding the missing terms does not by itself produce grasping at
100k, because the policy cannot discover a bonus it never reaches.** Anything
that closes this gap has to make contact reachable (a denser approach signal
that continues to the closing axis, a curriculum or demonstrations that put the
gripper on the piece, or longer training that happens to stumble into it), not
merely make contact valuable once achieved. Left default-off; no re-pilot
should adopt it on the strength of this run.

Two caveats on the comparison: episode length differs (300 staged vs 100
unstaged), so the staged policy had 3x more steps per episode to approach and
still did not do better; and the hold-at-goal risk above could not appear in a
run that never grasped.

## D21 — a pick-up training stage: grasp and lift 30 mm, with its own reward and observation, behind `pickup_mode` (default off)

**Date:** 2026-09-20 (`851e6f2`; probe configs `33569ca`).
**Numbering.** The roadmap table's numbering is canonical. This entry was
backfilled here from the commits; the repo's own D21 was the d1->a8
self-contact entry, which is now D24.

**Decision.** Add a separate training stage whose task is only to grasp one
piece and lift it 30 mm, as a stepping stone after D20's finding that contact
was not reachable from the staged shaped reward. Everything is config-gated and
default OFF, so the reward stack every earlier run used is unchanged (four
fixed-seed random traces, absolute and delta, with `shaped_staged` on and off,
plus a close-gated expert replay that grasps, all hash byte-identically to the
parent commit).

**What it pays.** `pickup_mode` replaces the reward outright (no goal-distance
term, no travelling failure penalty). Per step: a reach term times a posture
(tilt) factor, a close term gated on a viable pose ("pose gate") times gripper
closure, a lift term while HELD, a success term while succeeding, a one-time
first-grasp bonus, and a failure penalty on `bad_contact` / `piece_off_board`.

**Choices worth recording (each measured in the commit message):**
- Distance is taken to the grasp pose `piece + (0, 0, dz)`, not the piece
  origin. The e4->e5 diagnosis found the policy 45 deg off top-down at its
  closest approach. `dz` is `GRASP_Z` minus the piece's resting height, 0.03725
  +/- 0.5 mm.
- `POSTURE_FLOOR/SPAN` 0.5/0.5, not 0.25/0.75: at 0.25/0.75 the reward fell along
  the expert's own path (home 0.2373 vs pre-descend hover 0.2253).
- First-grasp bonus and lift term are keyed on HELD (both pads on the piece AND
  the piece clear of the board), not pinched and not `grasp_success`. Pinched
  alone paid for a non-grasp twice; D18's toppled piece is the other case.
- Success does not terminate the episode; `info["pickup_success"]` latches and
  reaches `monitor.monitor.csv` (the keyword is added only in pick-up mode).
- `pickup_obs` (default off, T1 only) appends 15 entries after the existing 21.

**Probes run** (`33569ca`, joint delta, e4->e5, 300-step budget, 1M steps: PPO
seeds 0/1/2, SAC seed 0). Counted from their `monitor.monitor.csv` on
2026-09-25: episodes with `pickup_success` true were 65/3408, 2/3358, 34/3381
and 18/3382; `terminated_reason == "success"` rows were 0, 0, 3, 1;
`solver_instability` rows 0 in all four. Not diagnosed here.

**Tests.** 37 new (148 -> 185), covering flag-off identity, each reward term,
the held-vs-pinched distinction against D18's toppled piece and a real expert
grasp, and per-episode state reset.

## D22 — a 4-dimensional Cartesian action mode (`action_mode="cartesian"`), with a height-dependent inner radial limit

**Date:** 2026-09-21 to 2026-09-22 (`c9bf295`, `f4cacdf`; merged 2026-09-25 as `8c11be2`).

**Decision.** Add a third action mode beside absolute and delta: `(dx, dy, dz,
gripper)`, driving the pinch point through mink IK with the tool locked
top-down to the pose D2's reach sweep validated. Absolute and delta are
untouched: byte-identical to `33569ca` on random-action and expert-replay
traces, pinned as frozen-hash regression tests.

**Design (`c9bf295`).** Each step rebuilds the target from the MEASURED pinch
position plus the scaled delta (never an accumulated target, D19's lesson),
clips it to a workspace box and then to a reach-aware radial limit, re-syncs
mink from the live qpos and solves. Points recorded with their measurements:
- Joint-change limiting is a uniform scale of the whole solved change
  (`_windup_scale`), not a per-joint clip: per-joint clipping rotated the tool
  out of top-down and then latched (48/60 steps rejected driving to e4,
  10.68 deg tilt); uniform scaling reaches the square in 56 steps, 0 rejects.
- The reach limit is height-aware (`CARTESIAN_REACH_Z/_R`); reach shrinks by
  115.6 mm from floor to box top.
- A rejected or non-converged solve increments `info["ik_rejects"]` (present
  and 0 in every mode) and holds the previous targets.
- `CARTESIAN_IK_MAX_ITERS` is 20 (the expert uses 400).

**Inner radial limit (`f4cacdf`).** `CARTESIAN_INNER_Z / CARTESIAN_INNER_R`
(margin 0.015 m) scales a target closer to the base axis than the limit
outward along its radius. The table comes from a static self-collision map of
`ur_upper_arm_link` against `ur_wrist_2_link`. At `GRASP_Z` (0.79925 m) the
limit is 0.2409 m, inside d1/e1 at 0.2501 m, so d1/e1 stay reachable; it does
not remove their self-contact (D24). Recorded limitation: a sustained push into
the near-base corner still produces self-contact (about 465 N peak over 250
steps plus a 20-step hold, 0 rejects), deferred to the referee's self-contact
policy.

**Measured.** All 64 squares reach hover and `GRASP_Z` (128/128, 0 rejects,
0.58-2.00 mm). IK cost, corrected in the project record: about 0.4 ms, one
iteration, about 10% throughput (an earlier figure of 5 ms / 11 iterations was
an artefact of driving a delta-mode env with absolute targets).

## D23 — Referee before training

Agreed 21 Sep. No training until the Stage 1 referee repairs (F1, F2, F7, F10 for T1, then F6 and F11 for T2) meet their exit evidence. Reason: the expert-replay gate uses the placement judge, F2 could pass it falsely, F1 hid table contact, and compute is scarce. HER stays off until F3 and F4 are repaired. Exit evidence met 25 Sep.

## D24 — self-contact on the d1/e1 starts: a hairline arm/wrist graze during `descend` (first recorded as `d1->a8`)

**Date:** 2026-09-22 (referee track, F1 round 2)
**Status:** open, not root-caused. No fix attempted; out of scope for F1
(`_bad_contact()`'s classification/ceiling work), which is what surfaced it.

**Finding.** F1 added zero-tolerance self-contact detection (no prior code
path could see robot self-contact at all). Running the scripted expert on
`d1->a8` now ends the episode via `bad_contact` at step 37, `descend`:
`ur_upper_arm_link` touching `ur_wrist_2_link`. With `_bad_contact()`'s
termination neutralized for measurement only (so the full trajectory is
visible), the contact persists for **11 consecutive macro-steps** and reaches
**0.1377 mm** of penetration at its deepest — a real, MuJoCo-confirmed overlap
(the model's global contact margin is 0.0, so this isn't a detection-tolerance
artifact), but a shallow graze, not a violent collision. The episode still
completes successfully once the gate is disabled: the arm brushes itself and
continues on to a normal placement.

**Isolated, not systemic.** The same measurement (real scripted expert, real
`_robot_geoms` ancestry classification, contact gate disabled) was run across
all 10 of the referee track's confirmed-clean regression pairs (`e4->e5`,
`a8->e5`, `h1->h8`, `a1->h8`, `d4->d5`, `c7->c6`, `b2->g7`, `f3->f6`,
`a4->h4`, `g1->b8`) plus `d1->a8`. Zero self-contact on all 10; `d1->a8` is
the only one of 11 showing any. This reads as an isolated scripted-expert
motion-planning defect for this specific pair's kinematic path — likely a
joint-space `descend` interpolation passing closer to the arm's own elbow/
wrist than intended, in the spirit of D10's joint-space-interpolation
mechanism, though that is not established here — not evidence that
zero-tolerance self-contact needs a general penetration-depth deadband
(the board-force ceiling's kind of decision). Left for a future session:
root-cause the `d1->a8` trajectory specifically, the same way D10 root-caused
`h1->h8`.

**Worth checking once training logs `terminated_reason`.** The scripted
expert is one fixed, open-loop trajectory per pair; an exploring RL policy is
not constrained to it and could plausibly produce brushing self-contact more
often than this single-pair, single-instance finding suggests. No training
data exists yet to check this against — flagged here so it's checked once
some does, not assumed either way.

**Update (2026-09-25) -- the finding was wider than "isolated".** Re-run on the
five-target 315-episode sweep, the upper-arm/wrist-2 contact fires on EVERY
trajectory that starts on d1 or e1: 10 of the 12 failures (d1 and e1 against
e5, a1, h8, a8, h1), each `bad_contact` at step 37 in `descend`, one macro-step,
on `ur_upper_arm_link` / `ur_wrist_2_link`. The other two failures are D10's.
The "one pair only" reading above came from an 11-episode sample that happened
to contain a single d1 start. Driven arrival at d1/e1 under Cartesian control
gives 65-68 N and 0.07-0.14 mm penetration. `CARTESIAN_INNER_R` is 0.2409 m at
`GRASP_Z` against d1/e1's 0.2501 m radius (D22), so the limit keeps the squares
reachable but does not prevent the contact. Still open: not root-caused, no fix
attempted. D25 does not touch this: arm-to-arm contact still terminates.

## D25 — gripper-internal contact is not a referee violation; closing on air is a recoverable action, judged by the success check and logged, not terminated

**Date:** 2026-09-25 (`5760449`).

**Finding.** With the F1 self-contact rule (D23), closing the gripper on
nothing put `ur_gripper_right_pad1` on `ur_gripper_left_pad1` and ended the
episode with `bad_contact` at step 9. Measured on current code: 4 of 4
close-on-air runs (delta and Cartesian, from home and hovering 0.09-0.10 m over
e4, piece elsewhere) terminated at step 9 with pad-on-pad forces of 2.75-8.3 N
and 0.01-0.03 mm penetration. Replays of the scripted expert through Cartesian
control (b4, g6, c7) showed the pads touching only 3-6 steps AFTER the piece was
lost, never while it was held; the joint-mode 315-episode expert sweep had zero
pad-on-pad steps. So the rule punished an empty close, not a bad grasp.

**Decision.** A robot-to-robot contact with one geom in the gripper's left
finger subtree (`ur_gripper_left_driver`, `..._left_spring_link` and below) and
the other in the right no longer counts toward `bad_contact`. 6 collidable
geoms per side, 36 exempt pairs; in practice pad on pad. MuJoCo's contact
physics is unchanged. Arm-to-arm, arm-to-gripper (including the gripper base
and mount), same-side finger and furniture contacts still terminate.
`info["gripper_self_contact"]` (bool, always present) reports the exempt
contact and is in `MONITOR_INFO_KEYWORDS`, so closing on air stays visible.

**Why a recoverable action.** An empty close is a mistake the task can recover
from (open again and retry) and is already judged by the success check;
terminating it also charges the failure penalty (771 at 300 steps, D20), which
would penalise closing the gripper at all; whether that changes learned
behaviour has not been measured.

**Verified.** Close-on-air x50 steps, four configurations: no termination, flag
true from step 9 (42/50 steps), aperture settles at 0.7822. D24 unchanged (d1/e1
still terminate). Negative controls that still terminate: arm-to-gripper
(upper arm through `gripper_base_mount`), pad against forearm, pad against
table. Full-board sweep 303/315, 0 of 315 episodes changed in outcome, reason,
length or contact pair. Throughput unchanged (561.5 vs 561.2 steps/s, noise
about 7%). 337 tests pass.

**Not addressed.** Why a held piece slips out during lift and place (b4, g6, c7
in the Cartesian replay) is a separate open question, not diagnosed here.

---

## D26 — `excluded_squares`: d1 and e1 are excluded from Cartesian-mode square sampling

**Date:** 2026-09-26
**Status:** accepted
**Evidence:** `results/d26_excluded_squares/` (determinism script, both hashes, README), branch `excluded-squares`
from `a42dfa7`. `tests/test_d26_excluded_squares.py`.

### Decision

New config key `excluded_squares` (list of square names, default empty). The rule is one sentence: an excluded square is
not used by the task in any role. It is removed, order-preserving, from the sampling pool
(`[s for s in ALL_SQUARES if s not in excluded]`), and a fixed start, target or obstacle list that names one is rejected
wherever it enters: the task constructors, per-episode `options`, and `resolve_config`. Validation lives in
`robochess/tasks/squares.py` (`validate_excluded_squares`, `reject_excluded`), shared as F11's obstacle validator is.
Use: `excluded_squares: [d1, e1]`.

### Reason

D24: under the zero-tolerance referee both d1 and e1 terminate on arm self-contact, in either action mode. Sampling
them puts episodes in the training set that cannot be completed.

### Cost

2 of 64 squares (62 remain). d1 and e1 are the queen and king home squares, so this is a documented limitation: the
benchmark does not cover starts or targets there.

### Default unchanged

With the default empty list the pool is the same 64 squares in the same order, so RNG draws are unchanged: 10,000
`(start, target)` pairs from seed 123 hash to `868586369d198f25602f4a1f2851b75ff878636befc2c659bc3969e5ca24f28e`
before and after.

### Side effect

Per-episode `options["start_square"]` / `["target_square"]` are now checked against the pool (previously only
start != target was checked when both were given). No existing test or literal in scripts passes a value outside
`ALL_SQUARES`; the only invalid literal, `z9`, is an already-expected constructor error.

## D27 — Placement requires carry and low release (H = 15 mm)

**Date:** 2026-09-26
**Status:** accepted
**Evidence:** `results/d27_release_latch/` (README names each file and how it was produced), branch
`referee-release-latch` from `ff7ea92`. `tests/test_d27_release_latch.py`.

### Problem

The adversarial sweep (branch `experiments/adv-sweep`, `a8517ba` and `eefd05e`) showed that F2 passes (a) a piece
pushed onto its square and then "released" with no grasp, and (b) a piece dropped from any height up to 81 mm.
F2 asks where the piece is and how it is resting; it never asks how it got there.

### Decision

Success gains an eighth term, `release_ok`, appended to the tuple `_success_terms()` returns; the other seven are
unchanged. `_is_success()` and `_build_info()` both require it.

* `ever_held` latches True on any step where `held` (both pads on the piece, no piece-board contact).
* The release record `(xy error, base height above the board)` is overwritten on every step where `pinched` (both
  pads on the piece), so it is the state at the last step the pads were on the piece. The height is the signed
  `piece_height_above_board`, the same expression F2's `resting` uses.
* `release_ok = ever_held and record exists and xy_error <= goal_tolerance (30 mm) and height <= H`.
* **H = `PLACEMENT_RELEASE_MAX_HEIGHT` = 15 mm**, locked before any result was seen.
* `info` gains `ever_held`, `release_height_mm`, `release_xy_error_mm` (both `None` until a record exists, not NaN,
  because `check_env`'s `data_equivalence` fails on NaN), `release_ok`; `ever_held` and `release_height_mm` are in
  `MONITOR_INFO_KEYWORDS`.
* `compute_reward` (the HER path) cannot recompute this history term; HER stays off (known issue F3).

`held` and `pinched` now exist in every mode: the one contact scan moved from the T1 task wrapper to
`RoboChessEnv._piece_contact_flags()`, run once per step (after `mj_forward` and the instability check, only when
not unstable) and cached. The wrapper's `_piece_contacts()` reads the cache, so `_grasp_success()` and
`PickupReward.contact_state()` are unchanged in classification (same bodies, same geoms). On an
`invalid_transition` step the cached flags, latch and record keep their last valid values (a behaviour change on
invalid transitions only). `reset()` scans the reset state, and the T1 and T2 `reset()` refresh again after they
move the piece or obstacles.

D27 only adds a condition that must also hold, so it can remove a success but never create one.

### Verified

* Suite: 362 passed (340 existing + 22 new), `-W error::UserWarning`, exit 0. Ten existing tests build success by
  teleporting the piece with no carry; each gained one `prime_carried_release(env)` line
  (`tests/_d27_helpers.py`) and no assertion changed. Two assertions on `MONITOR_INFO_KEYWORDS` changed to include
  the two new keys. The four trace hashes in `tests/test_cartesian_mode.py` are unchanged.
* Equivalence: on joint and Cartesian replays with `pickup_mode` on, the flags, D21 per-step rewards, `pinched`,
  `held` and the first-grasp step are identical to `ff7ea92` over 645 common steps (0 mismatches).
* Full-board expert (315): joint 303/315; T2 at `--obstacles 0` and `2` 303/315 each. 0 of 945 episodes differ from
  `ff7ea92` in success, outcome, phase or step count; the same 12 failures (10 d1/e1 `bad_contact` in descend, 2
  h1<->h8 `piece_off_board` in traverse). Release height over the successes: min -0.98, median -0.45, max 4.92 mm
  (joint).
* Cartesian v3 gate, 10 pairs: env `is_success` 10/10, identical step counts, release heights -0.30 to 9.70 mm.
* Throughput (bare env, random actions, 1 worker): 577.1 -> 558.7 steps/s median, about 3.2%.
* Adversarial sweep re-run (the sweep's own `section3.py`, plus two recovered scripts reproduced on `ff7ea92`):
  * 13 drops: the 5 at or below 15 mm (e4->e5 5 and 10, b7->g2 5 and 10, c1->f6 10) still succeed with episode
    length and D20/D21 identical to the sweep; the 8 above (20, 30, 80 mm) fail with `release_ok` as the only
    failing term. Recorded release heights equal the sweep's last-pad-contact heights to three decimals.
  * Push-then-open e4->e5 and b7->g2: `ever_held` False, `release_ok` the only failing term; the other six terms
    first all hold at step 150 and step 515, the sweep's F2 success steps.
  * Clean references c1->f6 and f1->c6: succeed, identical to the sweep.
  * b7->g2 topple: fails (final tilt 103.97 deg; resting, settled and upright terms), `ever_held` True,
    `release_ok` False (released at 22.2 mm).
  * The recovered scripts reproduced all ten of the sweep's recorded outcomes on `ff7ea92` before being run on D27.

### Fast set-down finding

Max-step descent plus a 2-step closed dwell topples the piece; max-step descent from 60 mm displaces it 16.4 mm and
tips it. D27 records the release correctly in all cases (`release_ok` True); the failures are F2's upright and XY
terms. This is a place-stage input, not a D27 issue. The test scenario has no dwell.

### Not addressed

The release record is sampled once per env step, like every other referee read (see F6); contacts inside a macro-step
are not seen. The sweep's substep measurements (`releases_2ms.json` in `eefd05e`) differ from the per-step figures by
up to 0.37 mm across the 13 drops, far from H, so no verdict here depends on it.

## D28 — Pick-and-place reward (`place_mode`, default off)

### Decision

`place_mode` (T1 only; needs `reward_type: shaped` and `pickup_obs`; rejected with `pickup_mode` and
`shaped_staged`) replaces the reward with `PlaceReward`. Design rule: finishing the move pays more than any way of
stalling or cheating, in total and discounted (gamma 0.99) return. D21's term functions are reused unchanged
(`pickup_reach`, `pickup_posture`, `pickup_pose_gate`, `pickup_closure`, `pickup_geometry`). Per step:

| state | condition | reward |
|---|---|---|
| S0 approach | not `ever_held` | D21 position + closing |
| S1 grasped | pinched and `ever_held` | 2.0 + 2.0 lift_frac + 2.0 closeness + 4.0 near (1 - lift_frac) |
| S2 placed | `release_ok`, not pinched, `piece_on_square`, piece tilt < 10 deg | 8.0 + 1.0 open_frac (max 9.0) |
| S3 lost | `ever_held`, not pinched, not S2 | D21 position + closing |

`open_frac = clip((0.782 - q) / (0.782 - 0.2), 0, 1)`, q the gripper qpos; 0.782 is the qpos fully closed on air
(measured; grasps on the stem sit at 0.647 to 0.651), 0.2 the F2 released threshold. One-time bonuses: +5 first held
step, +5 first 30 mm held 5 steps (replaces D21's +1 per step), +10 first entry to S2. Success (env `is_success`,
terminates) pays the constant `R_placed / (1 - gamma) + 20 = 1020`; `-5` on `bad_contact` / `piece_off_board`;
invalid-transition steps pay 0 and latch nothing (only the step counter, kept for info, moves). Config keys:
`place_mode`, `place_obs`, `place_carry_weight` 2.0, `place_lower_weight` 4.0, `place_placed_reward` 10.0,
`place_placed_floor` 8.0, `place_placed_ceiling` 9.0 (floor <= ceiling < R_placed), `place_placed_bonus` 10.0,
`place_lift_bonus` 5.0, `place_success_bonus` 20.0, `place_failure_penalty` 5.0. `place_obs` appends
`desired_goal - achieved_goal` (3 values, after the 15 `pickup_obs` values; `achieved_goal` is the piece body
position). S2's horizontal test is `info["piece_on_square"]` (goal_tolerance 0.03 m), the test success uses. The
15 mm in D27 is a release HEIGHT, not an XY limit. `monitor_info_keywords` adds `place_state` in place mode only.
Also added: `vecnormalize_clip_reward` (default 10.0, PPO's VecNormalize reward clip; every existing config unchanged).

### How the constants got here (each change approved before its results)

1. Original lump `R (1 - gamma^(T-t)) / (1 - gamma) + 20` with S2 paying a flat 10. Section 4 (Cartesian gate
   expert and Driver cheats, T = 300) passed the literal criteria, but two cheats never ran (success fires 4 to 5
   steps after the release, before a re-close or topple can act) and a **part-open parking exploit** was found:
   release to a gripper aperture of 0.4 to 0.6 (pads off the piece, above the 0.2 released threshold) and success
   never fires while S2 keeps paying 10 per step. Margin below the clean place, measured from step 0: 4.94, 1.91,
   2.28, 2.54 (e4->e5, a1->h8, c1->f6, b7->g2), against about 20 expected. Cause: everything is scaled by
   gamma^(t-1) at the success step (t = 171 to 252).
2. Rejected: hard-gating S2 on aperture < 0.2, which drops the reward from about 8 to S3 (<= 2) for the ~5 steps the
   gripper takes to open, punishing the release. Also rejected: scaling the success bonus (the margin is small
   because of where it is measured, not its size).
3. Adopted: the S2 ramp `8 + 2 open_frac`. Pre-registered expectation: parking margin from the release step at least
   3x the old value on every pair. **Failed under the like-for-like reading**: 2.98x, 2.24x, 2.42x, 2.53x versus the
   old reward measured from the release step (15.8x to 27.1x only against the old step-0 figures, which is the change
   of measuring point, not the ramp). Parked at aperture 0.203 the ramp adds almost nothing (margin 1.1x the old
   one), because S2 then pays 9.99.
4. Adopted after the audit below: the constant lump 1020 and the S2 cap of 9.0. Reason: both learners bootstrap the
   300-step truncation and the policy cannot see the step index, so the value of parking in S2 is `R / (1 - gamma)`,
   not a finite-horizon tail; a lump sized for a finite tail (about 27 discounted at the decision point) is what left
   parking so close. With the cap below R_placed, parking is worth at most 9 / (1 - gamma) = 900 against 1020 for
   finishing.
5. `vecnormalize_clip_reward`: PPO's VecNormalize divides the reward by the running return std sigma and clips at
   +-10. With sigma below about 75 the success lump (about 756 raw on e4->e5 under the original lump) is clipped to 10
   while per-step cheats are not, so staying in S2 without success or holding pinched on target would beat finishing
   under PPO. The place PPO configs set `vecnormalize_clip_reward: 1000`, which keeps the lump's ordering.

### Notes on the residual margins and a known limitation

* The re-grasp margin (9 to 10 from the release step) is the cost of delaying the constant lump by about one step
  (roughly gamma^n of 1020), not an exploit: re-grasping earns nothing extra and the S2 bonus fires once.
* Known limitation: S2 and the one-time bonuses depend on episode history (`ever_held`, `release_ok`, the first-held
  and first-S2 latches) that is not in the observation, as with D21's bonuses. The policy sees the current geometry and
  gripper state only, so the reward is not a function of the observation.

### Audit (before change 4)

* Truncation, SB3 2.9.0, script `results/d28_place_reward/audit_truncation.py` on a place-mode env with no gym
  TimeLimit in the stack: DummyVecEnv sets `info["TimeLimit.truncated"] = True` on the 300th step; PPO's
  `collect_rollouts` adds `gamma * V(terminal obs)` to that step's reward (buffer minus env reward is nonzero at steps
  300 and 600 only, and equals gamma V exactly); SAC's `DictReplayBuffer` has `handle_timeout_termination = True`,
  stores `timeouts` True at step 300 and samples that transition with `dones = 0`.
* Untrained policies (`untrained_policies.py`, 20 stochastic episodes each, e4->e5, 300 steps): freshly initialised PPO
  (with VecNormalize) and SAC occupy S0 for 100% of steps, never reach S1 or S2, invalid-transition rate 0; gripper
  qpos median 0.40 (PPO) and 0.39 (SAC), 6% and 4% of steps below 0.2; PPO 20/20 `time_limit`, SAC 16 `time_limit`
  and 4 `bad_contact`.
* Evaluation: `scripts/eval_probe.py` gains a place-mode report (`scripts/place_eval.py`): held (the D27 condition, both pads on the piece with no board contact; NOT D18 `grasp_success`), lifted 30 mm,
  carried within 3 cm, released, placed rates with Wilson intervals, the fraction of episodes truncated in S2 and their
  final gripper qpos, and a 20-episode deterministic run labelled effective n = 1 when the trajectories are identical.
  Output for pick-up runs is byte-identical to 76e116c (checked on a pick-up run); `--runs-dir` was added
  (default unchanged).

### Probe configs, smoke runs and the read-out rule

* `configs/_probe_cart_place_{ppo_seed0,ppo_seed1,ppo_seed2,sac_seed0}.yaml` are copies of the Cartesian pick-up probes
  (e4->e5, home start, 300 steps, 1M steps). Against their pick-up counterparts only these keys differ: `pickup_mode`
  becomes `place_mode`, `place_obs: true` is added, `vecnormalize_clip_reward: 1000` on the PPO files, `run_name`,
  `notes`. Training seeds 0, 1, 2 (PPO) and 0 (SAC); evaluation seed 20000.
* Checkpoints every 100,000 steps, each with its own VecNormalize file for PPO, so 500k and 1M are saved exactly (the
  pick-up runs did the same); a 5k smoke with a 2,500-step frequency saved 2,500 and 5,000.
* Smoke throughput, four concurrent runs on 12 cores: place PPO 157 to 159 fps against 181 to 186 for pick-up (about
  13 to 16% slower); SAC 32 fps either way. Estimated wall-clock for 1M steps: PPO about 1.8 h, SAC about 8.7 h. The
  Monitor CSV carries `place_state`. A 5-episode evaluation of each smoke model (seed 20000, frozen VecNormalize) ran
  end to end; the numbers (0/5 everywhere) only prove the pipeline.
* The decision rule is judged on `final_model` (1,001,472 steps for PPO), as for the pick-up probes; `rc_500000` is
  used only for the rising check.
* The grasp comparison with the pick-up probes (final: grasp 300/300 on all four; lifted 299, 299, 1, 300 of 300) uses
  D18 `grasp_ever` from `eval_probe.py`. The place report's "held" stage is a different metric. A grasp rate below 50%
  at 1M on two or more PPO seeds means D28 broke grasp learning, a different failure from not learning to place. The
  truncated-in-S2 tripwire (10% or more of evaluation episodes on any PPO seed) is a reported diagnostic; the S2 cap
  was its fallback and is already applied, so a hit is a finding, with no change mid-experiment.
* Two script fixes: `eval_probe_agg.py` prints the place section first and exits when the directory has no pick-up
  outputs (it previously crashed on the missing pick-up files before reaching it), and `place_eval.wilson` clamps to
  [0, 1]. Output for pick-up runs is byte-identical after both (aggregate output on the existing probe directory
  compared against 76e116c; `eval_probe.py` output on a pick-up run likewise).

### D27 push test is numeric-kernel dependent (found in CI)

`test_push_then_open_is_not_success_e4_e5` failed on CI for `place-reward` (a9a9191), with identical package versions
to a passing master rerun. It was not the branch: in an environment built with exactly the CI versions the test passes at
76e116c, 3aa7590 and a9a9191 with the default OpenBLAS kernel, and fails at 76e116c and a9a9191 alike (per-step traces
identical, 194 steps) when OpenBLAS is forced to a non-FMA kernel (`OPENBLAS_CORETYPE` SANDYBRIDGE, NEHALEM, PRESCOTT;
HASWELL and ZEN pass; SKYLAKEX could not be run on the local CPU). The traces first differ at step 12 in the fifth
decimal of the gripper qpos and grow into a different contact sequence. Under HASWELL the push pinches the piece at step
104 with the piece still on the board (not held). Under SANDYBRIDGE the piece is also pinched at steps 107 and 114 lifted
about 4 mm with no board contact, which is held; `ever_held` latches from a single such step.

* The pre-registered "a push fails with `ever_held` False" held locally but is kernel-dependent.
* The D27 guarantee held on every kernel tested: `release_ok` and `is_success` are False on every step of the push
  scenario. The test now asserts that guarantee on every step and records `ever_held` without asserting it.
* Full suites in the CI-version environment: 395 passed at 76e116c and 471 at 3aa7590 and a9a9191 (default kernel); under
  SANDYBRIDGE the only failure was this test, at 76e116c and a9a9191 alike.
* CI now prints `lscpu` (model name, flags) and `numpy.show_config()` before the tests, so the runner hardware is on record.

### Push-assisted diagnostic

D27 does not stop a policy from pushing the piece most of the way along the board and then taking one token lift to
latch `ever_held` (a held step while `release_ok` needs only a release at most 15 mm up on target). To see whether
successes are earned by a carry, the place evaluation reports, per episode, `held_steps` and `carry_mm` (the horizontal
distance the piece moved between its first and last held step), and for successful episodes the fraction whose
`carry_mm` is under 50% of the start-to-target distance, labelled "push-assisted". It is reported, not a gate: a real
carry over a short pair (e4->e5 is 78 mm) is fine, and the 50% line is a flag for reading the runs, not a criterion.
On the expert clean place e4->e5 it reads carry 73.6 mm of 78.0 mm over 74 held steps (not push-assisted).

### Acceptance (final constants; `results/d28_place_reward/acceptance.json`, `analyze.py`)

* 4a, the Cartesian gate expert on e4->e5, a1->h8, c1->f6, b7->g2: the recurring reward is non-decreasing through
  home, hover, close, first held, lifted 30 mm, mid-carry, over target at height, set down pinched, placed (S2 first step
  8.27 to 8.39, between 8 and 9); the S2 ramp is non-decreasing and at least 8 on every release. Values on e4->e5:
  0.239, 0.482, 1.892, 3.028, 4.867, 5.199, 5.898, 7.277, 8.273 (hand: about 0.24, -, 2, 2.88, 4.88, -, 6.0, 8.0, 9).
* 4b: every cheat is below the clean place in discounted return from step 0 and from the clean place's first release
  step, plain and bootstrapped (a case truncated at T gets `gamma^(T-t) * last reward / (1 - gamma)`), on all four
  pairs. Part-open parking margin from the release step, bootstrapped: 141.9 to 160.1 (aperture 0.40) and 112.2 to
  126.6 (aperture 0.203), against a required 100. Re-grasp margin from the release step is only 9.2 to 10.4. The
  fully-open topple and the lift-then-re-close never run (success first); they equal the clean place. Undiscounted
  return is informational: parking beats the clean place on e4->e5 only (+94, +139).
* Smoothness of the recurring reward on the four expert replays: median per-step change 0.012 to 0.022, 99th
  percentile 0.45 to 0.95, maximum 0.64 to 1.30 (the S1->S2 switch at release on three pairs, a place-descent step on
  c1->f6); largest change within a state 0.64 to 0.75. One-time events are +5, +5, +10 and the 1020 lump.

### Test-suite notes

* `tests/test_place_reward.py`: the invalid-step lift-streak assertion was corrected. Original: after four valid held
  steps and one invalid step ("would have been the 5th"), the next valid step was asserted to pay 0.0. That step is the
  5th valid one, so it correctly pays the +5; the assertion was wrong. Corrected: three valid steps, one invalid, then
  the 4th valid step pays 0.0 and the 5th pays 5.0.
* Building a real SB3 model inside the full pytest process segfaults torch (exit 139; the same test passes alone).
  A known test-environment issue: model-building checks run as scripts or in a subprocess
  (`audit_truncation.py`, `tests/test_place_eval.py`); `check_place_gamma` is unit-tested without a model.
* Counts: 443 (Section 3) + 7 (clip key) + 14 (S2 ramp; an earlier report said 21, a miscount) = 464; then +6
  (`tests/test_place_eval.py`) and a net +1 in `test_place_reward.py` = 471. In that file
  `test_lump_is_exactly_the_success_bonus_at_t_equals_T` was removed and
  `test_success_lump_is_the_discounted_tail_plus_bonus` (5 parametrised cases) became
  `test_success_lump_is_the_constant_value_of_staying_placed`; `test_lump_still_uses_r_placed_not_the_floor` became
  `test_lump_is_paid_on_top_of_a_floor_level_s2_step`; `test_lump_follows_r_placed_and_the_bonus_not_the_s2_ceiling` and
  `test_s2_ceiling_is_configurable_and_finishing_beats_parking` are new.
* Flag off: rewards, observations and info are byte-identical to 76e116c on a joint expert replay, a Cartesian
  replay and a pick-up replay (1128 steps, sha256 e4362657...da58 on both sides); the resolved config of each of the 29
  existing configs differs only in the new keys. `config_sha256` in a manifest hashes the config file bytes, and
  `run_name` comes from the filename and seed, so no run ID or hash changes; only `config_resolved` in new manifests
  carries the new keys.

### Read-out (probe results, e4->e5 Cartesian, 1M steps, eval seed 20000)

* Placement success (D27, `final_model`, 300 stochastic episodes, Wilson 95%): seed 0 297/300 = 99.0% [97.1, 99.7];
  seed 1 0/300 = 0.0% [0, 1.3]; seed 2 287/300 = 95.7% [92.7, 97.5]. Rule met on seeds 0 and 2, not on seed 1.
* Truncated-in-S2 tripwire (>= 10% of episodes on any PPO seed) did not fire at `final_model`: seed 0 0/300, seed 1
  0/300, seed 2 0/300 (seed 2's `rc_500000` reached 13/300 = 4.3%, still below the tripwire; the rule is judged on
  `final_model` only, as decided above).
* Push-assisted, among `final_model` successes: seed 0 0/297 = 0%, seed 2 0/287 = 0% (seed 1 has no successes).
* The D18 grasp comparison (`grasp_ever`; `_grasp_success`, `robochess/tasks/t1_pick_and_place.py:218-247`: piece
  origin more than `rest_tolerance` (10 mm) above the board, AND gripper qpos >= `gripper_open_threshold` (0.2), AND
  both pads on the piece, AND the piece not touching the board, all on the same step) FIRED on all three seeds at
  `final_model`: 46/300 = 15.3%, 40/300 = 13.3%, 137/300 = 45.7%, each below the 50% floor the diagnostic was written
  against, while the D27 "held" stage (pads on the piece, no board contact, no aperture or lift gate) is 99.7 to 100%
  and two of the three seeds place 95.7 to 99.0%. Recorded as a flawed pre-registered diagnostic: D18's simultaneous
  lift-and-closed-aperture-and-contact test, calibrated for the pick-up task, does not track grasp/hold behaviour in
  place mode, where the gripper carries the piece at a smaller aperture and lift than D18's floor.
* Seed 1 (0% placed) grasps near the start square and holds without carrying: 217/300 (72.3%) of episodes truncate
  in S1, final-step horizontal distance to target 73.2 to 108.8 mm (median 87.4 mm) against a 78.0 mm start-to-target
  distance on e4->e5, with 191/217 still held at the final step -- pinched and held without the piece having moved
  toward the target.
* SAC (0% placed at both `final_model` and `rc_500000`) lifts far past the 30 mm bar and holds there instead of
  carrying to the target: of the 282 (`final_model`) / 299 (`rc_500000`) episodes that truncate in S1, the max lift
  reached is a median 270.7 / 206.2 mm (up to 344.8 / 412.8 mm) and the final-step lift is a median 255.9 / 193.4 mm,
  while the final-step horizontal distance to target stays at a median 113.1 / 191.4 mm.
* Release height and XY error, among `final_model` successes (D27 `release_height_mm` / `release_xy_error_mm` at the
  final step): seed 0 (n=297) height 0.02 to 14.76 mm (median 7.08), XY error 10.38 to 29.96 mm (median 22.34); seed 2
  (n=287) height 0.04 to 13.56 mm (median 3.99), XY error 1.14 to 28.67 mm (median 14.30). Both within the 15 mm
  release-height ceiling and the 30 mm goal tolerance by construction (D27).
* Cross-check: `scripts/eval_probe.py` unmodified (300 stochastic, seed 20000, frozen VecNormalize) on seed 0
  `final_model` reproduces the custom evaluation exactly: placed 297/300, held 300/300, grasp 46/300, `rms_same` True.
* Underlying data, scripts and per-episode JSONs: `results/d28_place_reward/eval/` (untracked no longer; committed
  with this read-out).

### Diagnosis of non-placing runs (seed 1, SAC); no fixes applied

Read-only diagnosis, `results/d28_place_reward/diag/` (all scripts and JSON outputs committed with this section).
Method notes: action-distribution statistics are exact analytic quantities from the policy (Gaussian mean/std read
directly from `policy.get_distribution`/`actor.get_action_dist_params`, not sampled); counterfactuals fork the live
MuJoCo `data` plus every stateful attribute on the base env and `PlaceReward` via `copy.deepcopy`, verified to
produce an independent, steppable copy.

* **Seed 1**: at held steps the dz (dim 2) action's unclipped Gaussian mean is saturated past the -1 clip bound --
  final `median -1.93` (std 0.13, `P(|z|>1)` 0.96, i.e. lifting is effectively unsampleable); `rc_300000` `-0.83`
  (std 0.46, lift still occasionally sampled); `rc_500000` `-1.28` (std 0.27). The grip itself is adequate: forking
  10 stuck (step-150) episodes into a scripted lift-then-carry (the Cartesian gate expert's braked-step law) reaches
  within 3 cm and ends held in 9/10; seed 1's loss rate per held step (0.166/100) is the *lowest* of the three PPO
  seeds (seed 0 0.256, seed 2 0.412) once normalised for its ~250-vs-~75-step exposure. Contributing factor: the S1
  carry-reward gradient is 0.006 per mm at 78->60 mm from the target (seed 1's typical starting range) against
  0.12-0.19 per mm at 20->0 mm -- about 20x weaker far from the target, for both lift=0 and lift=20mm.
* **SAC**: not saturated -- dz pre-tanh mean median -0.08 (std 0.53) at held steps, i.e. centred near neutral with
  wide spread, unlike seed 1's locked-down mean. SAC instead holds the piece tilted: median 72 deg at step 150 (vs
  <=2.1 deg median, all three PPO seeds, at their first held step), pinched near its own base (pad height on piece
  median 2.4 mm). The S1 reward has no posture/tilt term, so a tilted hold at large lift still pays close to the S1
  ceiling (~4.7-6.6/step observed; matches the arithmetic 4.879 at lift>=30mm, d_xy=78mm) while placement additionally
  requires piece tilt < 10 deg (S2's own gate) -- unreachable from a 72-90 deg hold without first correcting the
  tilt, which nothing in the reward asks for. Height above 30 mm is unpaid: `lift_frac = clip(lift/0.03, 0, 1)`
  saturates, so the S1 reward is exactly flat (4.879 at this run's typical distance) for every lift from 30 to
  300 mm -- confirmed arithmetically, not just observed. A scripted move-above-then-descend-then-open counterfactual
  from 10 SAC stuck states set the piece down at 18-22 mm (never below), 82.5-90.9 deg tilt, with the Cartesian
  pinch site clamped at exactly `CARTESIAN_Z_FLOOR_ABOVE_BOARD` (31.5 mm) throughout; the naive
  floor-minus-pad-height arithmetic (31.5 - (-0.11) = 31.6 mm) does not match the observed 18-22 mm base height
  (off by ~11 mm), because the calibrated 37.25 mm vertical pinch-to-piece offset (`pickup_dz`) assumes an upright
  grasp and the piece is on its side. Seed 2's own successful releases sit above the floor (pinch z median 33.2 mm,
  pad height median 13.0 mm, tilt within the <10 deg gate) -- an upright hold clears the floor with room to spare,
  so the plateau is attributed to the tilt, not to the floor needing to move; **no environment change is indicated
  by this diagnosis**.
* **Reproducibility**: neither `scripts/eval_probe.py` nor `results/d28_place_reward/eval/custom_eval.py` seeds
  torch (`venv.seed()` only sets the env's own `np_random` for the next reset, not torch's global RNG that
  `predict(deterministic=False)` samples from) -- confirmed a fresh process's `torch.randn()` differs run to run.
  Stochastic evaluations in this project are therefore not bit-repeatable across separate invocations, though
  aggregate outcomes are often stable in practice because several action dimensions clip with high probability
  (Section 3c/4c). Not changed as part of this diagnosis.

## D29 — Posture gate, carry-potential shaping, PPO mean squash, eval seeding (each default off)

### Decision

Four independent additions on top of D28's `PlaceReward`, each behind its own config key, default off,
byte-identical to D28 when off (verified against a real `git worktree add d28-final`: 4 replay types -- joint
expert, Cartesian expert, each through `PickupReward` and `PlaceReward` -- sha256-identical on both sides; frozen
in `tests/test_cartesian_mode.py`'s idiom as `_D29_FLAG_OFF_HASHES`).

1. **Posture gate** (`place_posture_gate` false, `place_gate_tilt_lo_deg` 10.0, `place_gate_tilt_hi_deg` 45.0). In
   S1 only: `g = 1` at tilt <= lo, `g = 0` at tilt >= hi, `0.5 (1 + cos(pi (tilt - lo) / (hi - lo)))` between;
   `recurring = g * D28's S1 + (1 - g) * (D21 position + closing)`. `place_terms` gains `posture_gate` (always
   1.0 off S1 or with the flag off) and reuses the piece tilt D28's S2 gate already reads
   (`_piece_tilt_deg`). Validated: `lo < hi`; requires `place_mode`.
2. **Carry potential** (`place_carry_potential` false, `place_carry_potential_k` 50.0/m). `Phi(s) = -k * d_xy`, all
   states; `F = gamma * Phi(s') - Phi(s)` added to the reward on every valid step, including a terminal one (`Phi`
   at a terminal step is its actual post-step value, never zeroed -- the shaping is therefore not strictly
   policy-invariant; the D28 cheat ordering under it is re-verified empirically, not assumed, in 4b below). An
   invalid-transition step (F7): `F = 0`, the stored `Phi` is left at its last valid value, `place_terms["shaping"]`
   reports 0.0. `Phi` at reset is read from `_achieved_goal()`/`_desired_goal`, the same source
   `info["xy_distance_to_goal"]` is built from after the env's own F10 refresh. Validated: `k > 0`; requires
   `place_mode`. A stationary piece: `F = (1 - gamma) k d_xy` per step -- 0 at the target, otherwise a small
   positive leak (not exactly 0; exact only at `d_xy = 0`).
3. **PPO mean squash** (`ppo_mean_squash` false, PPO only -- rejected for any other algorithm).
   `robochess/policies.py`'s `SquashedMultiInputActorCriticPolicy(MultiInputActorCriticPolicy)` overrides only
   `_get_action_dist_from_latent` to `tanh` the Gaussian mean before building the distribution -- the one method
   `forward`/`predict`/`get_distribution`/`evaluate_actions` all route through, so training and eval agree. Noise
   (`log_std`), action clipping and the log-prob formula are unchanged. `build_model()` imports the module (and
   swaps `common["policy"]`) only when the flag is on; off, `policy="MultiInputPolicy"` stays the literal string
   D28 used, byte-identical. `scripts/eval_probe.py` imports the class unconditionally at module scope so
   `PPO.load()`'s cloudpickle-by-reference resolution always finds it.
4. **Eval seeding**. `scripts/eval_probe.py` calls `stable_baselines3.common.utils.set_random_seed(seed)` then
   `model.set_random_seed(seed)` right after `.load()`, before the existing `venv.seed(seed)` (SB3 2.9.0's
   `set_random_seed` also calls `self.env.seed(seed)` when `self.env` is set, so this is a harmless double-seed of
   the env, kept for a minimal diff). This can change a stochastic (default) eval's action sequence relative to an
   unseeded run at the same `--seed`, since it now also seeds the policy's own torch RNG; deterministic evals are
   unaffected (no sampling). Old stochastic-eval outputs were not regenerated. The env's own first-reset observation
   for a given seed is unchanged by this (verified: built the real pipeline twice, with and without the two new
   calls, identical `venv.reset()` output both times). `place_eval.py` needed no change -- it never loads a model or
   samples an action, confirmed by inspection (pure functions over already-recorded rows).

`place_terms` always carries `posture_gate` and `shaping`, flag on or off, so downstream code needs no conditional
key access.

### Rule 4 diagnostics gained two fields (report only, not gated)

`scripts/eval_probe.py`'s place-mode rows gained `pg` (`place_terms["posture_gate"]`); `place_eval.py`'s
`place_episode_summary` gained `s1_steps`, `s1_gate_lt1_steps`, `max_s1_tilt`; a new `gate_diagnostics()`, wired
into `place_report()["gate"]` alongside `push_assisted`, reports -- over PLACED episodes only -- the fraction of
S1 steps the gate discounted (`posture_gate < 1`) and the max S1 tilt reached. Both are 0.0/None with the gate off
or no placed episodes.

### Pre-launch report-only replay of D28 PPO seeds 0 and 2

50 episodes each, seed 20000, stochastic, torch-seeded, frozen VecNormalize, e4->e5. Seed 0: median S1 tilt 2.0
deg, p90 4.7 deg, max 125.1 deg, 6.5% of S1 steps > 10 deg, 4.0% > 20 deg, mean gate D29 would assign 0.960. Seed
2: median 6.4 deg, p90 66.7 deg, max 80.5 deg, 43.1% > 10 deg, 26.6% > 20 deg, mean gate 0.777. Split by episode
outcome and by phase (before the first loss of pinch / after a re-grip): in **placed** episodes, seed 2 carries at
tilt > 27.5 deg on 19.8% of S1 steps (max 80.45 deg) and always releases below 10 deg; seed 0 on 0.0%. Neither
seed's not-placed episodes ever reach a re-grip (0 after-loss steps -- they get stuck on the first attempt).

Tilt alone cannot separate these carries from SAC's toppled hold (D28 diagnosis: 72 to 90 deg). Gate kept at 10/45
by decision; the gate discounts swinging carries by design. A persistence- or grip-height-based gate is the
candidate if D29 PPO fails rule 1.

### Acceptance (pre-registered; `results/d29_place_reward_v2/acceptance.py`, `analyze.py`, `acceptance.json`)

**ALL PRE-REGISTERED CHECKS PASS.** Gate and potential on throughout (defaults: tilt 10/45 deg, k 50/m).

* 4a, Cartesian gate expert, e4->e5 / a1->h8 / c1->f6 / b7->g2, gate and potential on: recurring reward
  non-decreasing through all nine checkpoints on every pair, S2 non-decreasing and >= 8, placed in [8, 9]; state
  reward equals the D28 value EXACTLY (diff 0.0, not just within tolerance) at every S1 step with tilt <= 10 deg
  (56 to 138 such steps per pair); max per-step reward change attributable to the gate is 0.0 on every pair -- the
  gate never actually engages on a clean expert path (tilt stays under 1 deg throughout); shaping totals per pair
  +7.5 to +94.0.
* 4b, every D28 cheat plus `shaping_farm` (oscillate toward/away from the target while held, to timeout) and
  `toppled_hold` (e4->e5 only -- see below) below the clean place in all four discounted-return comparisons
  (step-0, from-release, both bootstrapped at truncation) on every pair, 48/48 judged cases "ok"; the two literal
  cheats (`release_then_topple_open_gripper`, `release_lift_reclose_literal`) remain n/a, success still fires
  before the mechanism runs; `push_then_grasp_release_on_target` (push most of the way, then a genuine grasp and
  place) reported only, not judged. Parking margin from the release step, bootstrapped: 8/8 pass (>= 100 required;
  actual 112.1 to 160.1, matching D28's own figures, confirming the new terms did not disturb this margin).
* **`toppled_hold`** (e4->e5 only): rather than script an in-grasp topple -- four maneuvers tried (violent lateral
  shake while gripped, a sideways drag at grasp height, a shake with a loosened grip, slamming straight down into
  the board while gripped) all failed; the stem-clamp grasp is either too rigid (tilt stayed under 8 deg) or drops
  the piece the moment it loosens -- this cheat replays the exact 150 actions recorded from a real D28 SAC seed0
  episode (seed 20000, stochastic, torch-seeded; found on the very first of up to 20 scanned episodes: pinched,
  tilt 90.61 deg at step 150) through the D29-flagged env, confirmed bit-identical to the original physics at
  t=150 (piece xyz max abs diff 0.0 m, tilt identical to full float precision -- reward doesn't touch physics),
  then holds (zero displacement, closed gripper) to timeout. 240 pinched steps, 226 with `posture_gate < 1`, 196
  with tilt >= 45 deg; mean D28-equivalent S1 payout over those 196 steps 4.03/step (max 4.21), mean D29 payout
  0.78/step.
* **4c**: S1 reward <= 2.2 whenever tilt >= 45 deg -- PASS, 0 violations of 196 qualifying steps in the
  `toppled_hold` replay (actual max paid 0.77).
* Scripts and artifacts committed: `acceptance.py`, `analyze.py`, `acceptance.json`, `analyze_output.txt`,
  `find_topple_episode.py` (the SAC-episode search, its own process -- building an SB3 model alongside
  `acceptance.py`'s many `RoboChessEnv` instances in one process risks the same intermittent segfault noted in
  D28's test-suite notes), `topple_actions.json` (its output, replayed with no model needed),
  `tilt_distribution.py` / `tilt_distribution_detailed.py` and their JSON outputs.

### Test-suite notes

* `tests/test_d29_place_reward_v2.py` (41 tests) plus 4 new tests in `tests/test_place_eval.py`: gate values at
  the pre-registered tilts (continuity confirmed exactly at tilt 10 and 45 deg), S1 equals D28 when the gate is off
  or tilt <= 10, gate-on/tilt>=45 equals `position + closing` to 1e-12; shaping sign and telescoping-identity tests
  (`sum(gamma^i F_i) == gamma^n Phi(final) - Phi(initial)` to 1e-9 on a zero-invalid-step replay, and separately on
  a real Cartesian-expert replay: n=171, invalid=0, diff 2.665e-15); invalid-step F=0/Phi-held; the closed-form
  stationary-piece check; mean-squash bounds for extreme latents via both `get_distribution` and
  `evaluate_actions`; save/load round-trip and an end-to-end `eval_probe.py` subprocess run on a fabricated
  `tmp_path` run (never the real `runs/`); the two eval-seeding tests (`skipif` when
  `runs/_probe_cart_place_ppo_seed0` is absent -- verified by moving it aside, both tests skip cleanly, then
  restoring it, both pass); the gamma guard (`check_place_gamma` direct calls, and a monkeypatched spy through a
  real `build_model()` call); one test per config-validation branch; the 4 frozen flag-off hashes.
* Same segfault risk D28's own notes flag (building a real SB3 model inside the full pytest process segfaults
  torch intermittently, exit 139): three tests that originally built or loaded a model in-process were rewritten
  to do it in a throwaway subprocess (`subprocess.run([sys.executable, "-c", code], ...)`, mirroring
  `test_place_eval.py`'s `_make_run`) -- the gamma-guard build test, the mean-squash build/save/reload test, and
  the eval-seeding first-reset-obs test (`PPO.load()` also reconstructs a model).
* Full suite: 517 passed (475 pre-existing + 41 new in `test_d29_place_reward_v2.py` + 1 net in
  `test_place_eval.py`), 0 failures, `-W error::UserWarning`, exit status 0.
* CI failed on exactly the 4 frozen flag-off hash tests: CI's CPU produces different contact-rich expert
  trajectories (the joint expert logs a 1.79 mm pinch residual there) from this machine's, the same numeric-kernel
  sensitivity D28's own test-suite notes recorded for the D27 push test. The flag-off identity itself was proven
  locally against a real `d28-final` worktree (see the Decision section above); the 4 tests now skip when
  `CI` is set in the environment, with the frozen hash values left unchanged.

### Probe configs and smoke

* `configs/_probe_cart_place29_{ppo_seed0..4, ppotanh_seed0..4, sac_seed0}.yaml` (11 files): copies of their D28
  counterparts (`_probe_cart_place_ppo_seed{0,1,2}`/`_sac_seed0` for seeds 0-2/SAC; D28 had no seed 3/4, so those
  are copies of `_probe_cart_place_ppo_seed0` with only `seed` differing besides the keys below) with
  `place_posture_gate: true`, `place_carry_potential: true` (and `ppo_mean_squash: true` for the ppotanh files)
  added; verified programmatically (resolved-config dict diff) that no other key differs besides `seed`, `run_name`,
  `notes`. `vecnormalize_clip_reward` stays 1000 in every PPO config (confirmed).
* All 11 5k-step smokes run concurrently (12 cores): 8/11 succeeded immediately; 3 hit a transient
  `torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable` under the 11-way concurrent launch
  despite `device: cpu` in every config -- a real GPU is present (RTX 2060 via WSL2 passthrough), and its driver
  transiently refused context creation for 3 of the 11 simultaneous processes. Retried those 3 sequentially with
  `CUDA_VISIBLE_DEVICES=""` and they completed cleanly; all 11 have a `final_model.zip`. **Launch note: concurrent
  launches hit a transient CUDA context error despite `device: cpu`; all D29 runs launch with
  `CUDA_VISIBLE_DEVICES=""` (CPU-only, no effect on results).**
* fps (genuinely concurrent runs only, excluding the 3 retried sequentially): PPO/ppotanh 123 to 129 (mean 126),
  SAC 31. Estimated wall-clock for the full 11-run batch at 1M steps, set by the slowest: PPO-family ~2.3 h, SAC
  ~9.0 h -- **~9 hours overall**, dominated by SAC. Monitor CSV carries `place_state` (confirmed in a PPO, ppotanh
  and SAC smoke run's header). `place_eval` ran end to end on the `ppotanh_seed0` smoke model (5 episodes,
  1000-model-load subprocess via `eval_probe.py`): no errors, confirms `PPO.load()` resolves
  `SquashedMultiInputActorCriticPolicy` from the saved zip in a fresh process; 0/5 placed (5k steps, expected) with
  the new `gate` diagnostic correctly null when no episode is placed.
* Smoke run directories (`runs/_smoke29_*`) are not committed (gitignored, as with every prior smoke run).

### Pre-registered rules for the full launch (recorded verbatim before any 1M-step run)

1. Each PPO variant: placement >= 10% on >= 4 of 5 seeds at `final_model` (300 stochastic, seed 20000, torch-seeded,
   frozen VecNormalize).
2. The benchmark adopts `ppo_mean_squash` only if it passes rule 1 on more seeds than PPO default; ties keep SB3
   defaults.
3. SAC: reported, no rule (n = 1). **Expected (a lead, not a rule):** SAC's toppled-hold fraction (tilt > 45 deg
   while pinched at step 150) falls from its D28 level.
4. Diagnostics, reported not gated: saturation (fraction of held steps with `P(clipped) > 0.9` on dx, dy or dz),
   gripper command < 0.9 while held, truncated in S2 (tripwire >= 10%), push-assisted fraction, toppled-hold
   fraction, shaping total per episode, and (D29) per seed the fraction of S1 steps with `posture_gate < 1` in
   placed episodes and the max S1 tilt in placed episodes.
5. Then widen to random squares with the adopted PPO variant; that rule is pre-registered before launch.

### Read-out (rules 1 and 2, SAC, failure taxonomy)

300 stochastic episodes per `final_model`, seed 20000, torch-seeded, frozen VecNormalize (PPO; SAC carries
none). Full detail, every script, and every number's provenance: `results/d29_place_reward_v2/eval/README.md`
and `results/d29_place_reward_v2/eval/section6_summary.md`.

* **Rule 1**: default **FAILS (3/5 seeds pass)** -- seeds 2, 3, 4 pass (96.0%, 97.7%, 98.0%); seeds 0 and 1
  are stuck at 0.0%. Squash **PASSES (4/5 seeds pass)** -- seeds 0, 2, 3, 4 pass (92.3%, 96.0%, 99.3%,
  97.3%); seed 1 is stuck at 0.0%. All ten Wilson 95% CIs in the summary linked above.
* **Rule 2**: squash **ADOPTED**, on a one-seed margin (4/5 vs 3/5) decided by seed 0 alone. Fisher exact
  test on the 4/5-vs-3/5 table: **p = 1.0** (not significant) -- among seeds passing on both variants (2, 3,
  4), placement rates are close (e.g. seed 2: 96.0% vs 96.0%). This is adoption under the pre-registered
  rule, not evidence squash performs better than default PPO. `ppo_mean_squash`
  (`robochess/policies.py`'s `SquashedMultiInputActorCriticPolicy`) is accordingly a **documented deviation
  from SB3 defaults**: it tanh's the Gaussian mean before building the action distribution, in every
  codepath that builds one.
* **SAC** (reported, no rule): **0/300 placed**. Toppled-hold fraction (tilt > 45 deg while pinched at step
  150) **94.0% to 0.0%** with the posture gate and carry potential both on (D28 SAC seed0, gate+potential
  off: 282/300; D29 SAC seed0, gate+potential on: 0/300 -- one seed each, consistent with the posture gate,
  not attributed to it as sole cause on this n=1 comparison). SAC's failure is carrying and descending, not
  releasing: never within 3 cm of the target across all 300 episodes; S1 medians lift 110.3 mm, xy-distance
  45.9 mm (D28 baseline, re-measured with the same methodology: lift 244.7 mm, xy-distance 119.7 mm).
* **Failure taxonomy**:
  - Default s0 holds 19 mm from target, upright and low, and never releases -- a **local optimum, not a
    reward loophole** (its real discounted return, G0 mean 234, is well below a real placement's, G0 mean
    555; from a held-within-30mm step, 434 vs ~1015).
  - Default s1 grasps and never carries -- holds 124 mm from the target.
  - Squash s1 carries the piece the full distance and places it correctly (all 6 of 7 success terms true at
    the final step in all 87 of its S2-ending episodes) but never opens the gripper past the release
    threshold (`gripper_released` False in 87/87). A tip-grip lead (elevated grip position on the piece) was
    tested and **not confirmed** -- squash s1's pad-contact height while held (31.4 mm) is normal, matching
    default s3 (29.4 mm) and squash s3 (30.5 mm); it is squash s0, a *placing* seed, whose held pad height
    is elevated (57.9 mm, 7.9 mm above the piece's own 50 mm total height).
  - SAC holds the piece 110 mm high and 46 mm from the target and never descends toward it.
* **Open items**: squash seed 0's training-log/evaluation gap (97.4% training-log last-500 vs 92.3%
  evaluation, unexplained); the tip-grip lead for squash s1 (unconfirmed, see taxonomy); squash s1 holds at
  qpos 0.775 (closed-on-air is 0.782) at a normal pad height -- unexplained, parked; seed 1 fails across
  every variant measured (D28 default, D29 default, D29 squash), each a different failure mode; SAC's next
  diagnostic step (observation-only VecNormalize, untested -- SAC's config currently carries none).

ROADMAP.md sync pending (separate docs commit).

## D30: T1 benchmark task is the fixed move e4->e5 (4 Oct 2026)
- Context: RS29 (pre-registered, results/rs29_random_squares/) trained PPO with mean
  squash (D29) on uniform random start/target pairs (d1, e1 excluded, D26), 400-step
  budget, 3M steps, 6 seeds. Result: 0 of 6 seeds learn; 0/300 placements per seed on
  the 20 fixed evaluation pairs. Positive control (D29 squash seed 3 on e4:e5 at
  1c215fe) placed 298/300, so the code is not at fault.
- Diagnostics: holds are brief and slip (median 2-6 steps held in 5 of 6 seeds), no
  net progress toward targets, failure on short pairs too (so not distance), grasping
  degraded with more training in seeds 1 and 4. Exploratory zero-shot: D29 squash
  models place only on e4:e5 (15/15) and on no other pair; the fixed-move policy does
  not transfer to other start squares.
- Decision: the T1 benchmark task is the fixed move e4->e5 with the D29 squash
  configuration (configs/_probe_cart_place29_ppotanh_seed*.yaml settings). T2 and T3
  will also use fixed configurations, specified in their own specs. Random-square
  generalisation is reported as a pre-registered negative result and a documented
  limitation of the benchmark. A start-square curriculum is future work. Warm-starting
  from a PPO policy is rejected for the factorial because SAC cells could not use it.

## D31: The D recipe is the default PPO recipe (16 Oct 2026)
- Recipe: n_envs 4, ppo_lr_schedule linear (3e-4 decayed to 0), ppo_ent_coef 0.01; SB3 defaults
  otherwise (n_steps 2048 per env, batch 64, 300-step episodes).
- Evidence: RS31 (pre-registered, results/rs31_24_pairs/) compared arm C (ent_coef 0.01, n_envs 1,
  constant lr 3e-4) with arm D (C + n_envs 4 + linear lr decay), seeds 0 and 1 per arm, 3M steps, 24
  start squares: 2 of 2 D seeds learned, 0 of 2 C seeds (results/rs31_24_pairs/eval/README.md, rule 2).
- Caveats: (1) the recipe is a bundle; n_envs 4 and the linear decay changed together, so no
  attribution to either is claimed, and ent_coef 0.01 was common to both RS31 arms. (2) Two seeds per
  arm. (3) PPO only: SAC is unaffected (n_envs > 1 and ppo_* keys are rejected for SAC). (4) RS32 then
  ran the D recipe on 48 start-target pairs, 6 seeds, 5M steps: 0 of 6 seeds meet the pre-registered
  rule (results/rs32_48_pairs/eval/), so the recipe is the default for the fixed-cell benchmark and is
  not evidence of multi-start generalisation beyond the RS31 24-pair result.
- Decision: new PPO cells use the D recipe unless a pre-registration says otherwise. Existing
  configs are unchanged (the schema defaults are not edited: n_envs 1, constant lr, ent_coef 0.0 remain
  the resolve_config defaults, so old manifests still resolve to what they trained on).
