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
session); root-caused 2026-09-10 (kinematic replay, same session)
**Status:** **root-caused — mechanism confirmed.** Fix not yet designed, not
attempted.

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
also fails `h8->h1`, the same file traversed in the opposite direction. It is
classified as this entry's mechanism rather than D11's. Reversing the
direction of travel does not avoid the arc, which is what the mechanism
predicts — the joint-space path between the same two widely-separated poses
is the same curve traversed either way. No per-episode trace for `h8->h1` is
reproduced here; it is counted against the `h1` target in the per-target
table below, since that is the square it places on.

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

**Fix not yet designed.** Per this project's diagnose/fix separation, no fix
was attempted in the session that root-caused this. The two obvious
directions — subdividing `traverse` into task-space waypoints so the
commanded route stays near the Cartesian straight line, and reconsidering
whether `_piece_off_board()` should fire on a securely gripped piece in
mid-transit at all — are candidates only; neither has been designed, costed
or measured.

**Consequence:** not a blocker for D8/D9 — both are release-phase fixes and
this occurs upstream of release. Does need fixing before leaning on a
"100% board coverage" claim anywhere, and the cluster above says the margin
on the passing cases is thin enough that a change of transit height or board
placement could put more episodes over the bound.

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

**Consequence:** not a blocker for D8/D9, which are release-phase fixes; all
four instances lose the piece before `release` does anything. Does need
root-causing before any "100% board coverage" claim.
