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
