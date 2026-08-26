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

**Date:** 2026-08-27
**Status:** diagnosed, not fixed. Fix design is separate, upcoming work —
not attempted here, per this project's diagnose-then-fix convention.
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

### Consequence

Not attempted here, by the project's diagnose-then-fix convention.
Candidate directions for a future fix-design session, not evaluated here:
damping the release open-command (a velocity ramp, analogous to D7's
rejected closing-side attempt, though D7's attempt failed for the closing
case for reasons that may or may not transfer to opening); a grasp geometry
with more rotational constraint than a pure head contact (revisits the same
base/stem/head tradeoff D7 already investigated); or a settle-and-check
step after release before declaring the waypoint sequence finished, so a
piece caught mid-spin gets more time rather than being scored against
whatever state the episode happens to end in. All of these need actual
verification before adoption, per this project's standing lesson that a
plausible-sounding fix must be measured, not inferred.
