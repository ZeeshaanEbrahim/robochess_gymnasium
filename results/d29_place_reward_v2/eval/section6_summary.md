# D29 place-reward-v2 eval summary (results/d29_place_reward_v2/eval/section6_summary.md)

Runs trained at 41ae8e5; evaluation committed at c4032a8. Branch `place-reward-v2`. 11 runs (10 PPO, 1 SAC),
300 stochastic episodes each, seed 20000, torch-seeded, frozen VecNormalize (PPO only; SAC carries no
VecNormalize). All numbers below are drawn from the eval outputs already produced in this directory; none
are new claims.

## (a) Rule 1

Rule 1: a PPO variant learns T1 e4->e5 reliably if placement is at least 10% on at least 4 of 5 seeds at
`final_model` (300 stochastic episodes, seed 20000, torch-seeded, frozen VecNormalize).

| variant | seed | placed k/n | rate | Wilson 95% CI | >= 10%? |
|---|---|---|---|---|---|
| default | 0 | 0/300 | 0.0% | [0.0%, 1.3%] | No |
| default | 1 | 0/300 | 0.0% | [0.0%, 1.3%] | No |
| default | 2 | 288/300 | 96.0% | [93.1%, 97.7%] | Yes |
| default | 3 | 293/300 | 97.7% | [95.3%, 98.9%] | Yes |
| default | 4 | 294/300 | 98.0% | [95.7%, 99.1%] | Yes |
| squash | 0 | 277/300 | 92.3% | [88.8%, 94.8%] | Yes |
| squash | 1 | 0/300 | 0.0% | [0.0%, 1.3%] | No |
| squash | 2 | 288/300 | 96.0% | [93.1%, 97.7%] | Yes |
| squash | 3 | 298/300 | 99.3% | [97.6%, 99.8%] | Yes |
| squash | 4 | 292/300 | 97.3% | [94.8%, 98.6%] | Yes |

**Default fails rule 1 (3/5 seeds pass). Squash passes rule 1 (4/5 seeds pass).**

## (b) Rule 2

Rule 2: the benchmark adopts `ppo_mean_squash` only if it passes rule 1 on more seeds than PPO default; ties
keep SB3 defaults.

**Squash is adopted** on a one-seed margin (4/5 vs 3/5) decided by seed 0 alone (squash seed 0 passes at
92.3%; default's non-passing seeds 0 and 1 are stuck at 0.0%, not near the 10% line). Fisher exact test on
the 4/5-vs-3/5 contingency table gives **p = 1.0000** (not significant). Among seeds that pass on both
variants (2, 3, 4), placement rates are similar between default and squash (seed 2: 96.0% vs 96.0%; seed 3:
97.7% vs 99.3%; seed 4: 98.0% vs 97.3%). This is adoption under the pre-registered rule, not evidence that
squash performs better than default PPO.

## (c) SAC (reported, no rule)

SAC seed 0, 300 stochastic episodes: **placed 0/300 (0.0%, Wilson [0.0%, 1.3%])**. Held 300/300 (100%),
lifted_30mm 300/300 (100%), carried_within_3cm 0/300 (0.0%), released 0/300 (0.0%).

Toppled-hold fraction (tilt > 45 deg while pinched at step 150): **94.0% to 0.0%** with the posture gate and
carry potential both on, one seed each (D28 SAC seed0, gate+potential off: 282/300; D29 SAC seed0, gate+
potential on: 0/300) -- consistent with the posture gate, not attributed to it as sole cause on this n=1
comparison.

SAC's own S1 medians (in-grasp, D29 vs the freshly re-measured D28 baseline, same run and step-150/tilt-45
methodology): D29 lift 110.3 mm / xy-distance 45.9 mm / tilt 0.2 deg; D28 lift 244.7 mm / xy-distance 119.7
mm / tilt 72.6 deg (D28's own earlier readout cited a median hold ~113 mm from the target -- in the same
range as this xy-distance re-measurement, not identical, likely a different aggregation). SAC is never
within 3 cm of the target across all 300 episodes; its failure is carrying and descending, not releasing.

## (d) Failure taxonomy

- **Default s0**: holds 19 mm from target, upright and low, and never releases. A **local optimum, not a
  reward loophole**: its actual discounted return (G0 mean 234, from a held-within-30mm step mean 434) is
  well below a real placement's (default s3 placed G0 mean 555, from-held mean 1013; squash s3 similar).
- **Default s1**: grasps and never carries -- holds 124 mm from the target (far outside the goal tolerance)
  and does not close the distance.
- **Squash s1**: carries the piece the full distance and sets it down on square, upright and settled (all 87
  S2-ending episodes), but never opens the gripper past the release threshold (`gripper_released` False in
  87/87); success therefore never fires. Tip-grip lead (holds at a different qpos than a normal grasp) is
  **unconfirmed** -- see open items.
- **SAC**: holds the piece 110 mm high and 46 mm from the target and never descends toward it; the posture
  gate eliminated its D28-era toppling (94.0% to 0.0%) without producing any placements.

## (e) Open items

- Squash seed 0's training-log/evaluation gap (97.4% training-log last-500 vs 92.3% evaluation) -- parked as
  unexplained; does not affect rule 1 (squash seed 0 still passes at 92.3%).
- The tip-grip lead for squash s1 -- tested and **not confirmed**: squash s1's held qpos (0.775) is well
  above default s3's (0.546) and squash s3's (0.546), matching the original observation, but the measured
  pinch-site height above the piece base (`ur_gripper_pinch` site z minus piece body origin z; not a
  pad-contact measurement) while held is **not** elevated for squash s1 (median 31.4 mm) -- it is close to
  default s3 (29.4 mm) and squash s3 (30.5 mm). It is **squash s0** (a placing seed) whose held pinch-site
  height is elevated (57.9 mm), the opposite of what the lead predicted. The qpos difference is real but not
  explained by pinch-site height on the piece.
  - **(i) Pad-height definition, for reconciling against D28's ~13 mm figure.** This pass measured
    `data.site_xpos[pinch_site]_z - data.xpos[piece_body]_z`, i.e. the fixed `ur_gripper_pinch` site (the
    same reference point the reward/geometry code uses throughout, `robochess/tasks/t1_pick_and_place.py`'s
    `_pinch_site_id`) minus the piece body's own origin, which sits at the piece's base (local z=0; the
    piece's `freejoint` body origin, `assets/scene.xml:141-148`). This is a fixed gripper-frame reference
    point, not a measurement of where the pad surfaces physically touch the piece -- the two need not
    coincide. The piece's total height, from the same geometry (base cylinder + stem cylinder + head
    capsule, `assets/scene.xml:143-145`): 0 to 0.050 m (50 mm) above its own base. Squash s0's 57.9 mm
    is **not on the piece** -- it is 7.9 mm above the piece's own physical top, confirming the pinch site
    is offset from the piece surface for this seed's grip, not a literal pad-contact depth. D28's ~13 mm
    figure was not something this pass could locate in committed D28 outputs (checked `DECISIONS.md`, the
    D28 diag/eval JSONs); it may use a different reference point (e.g. an actual contact-solver position
    rather than the fixed pinch site), so the two numbers are not yet known to be comparable.
  - **(ii) Squash s1 holds at qpos 0.775 (closed-on-air is 0.782) at a normal pinch-site height (31.4 mm,
    not elevated).** So it grips almost as tightly as physically possible, at a normal position on the
    piece. Unexplained; parked.
- Seed 1 fails across every variant measured so far: D28 default PPO seed 1, D29 default PPO seed 1, and D29
  squash seed 1 (a different failure mode in each case -- see taxonomy above and the section 4 follow-up
  trap-state analysis).
- SAC's next diagnostic step: observation-only VecNormalize (SAC's config carries none; whether adding one
  changes the carry/descend failure is untested).

## (f) Deviations from SB3 defaults

- **PPO mean squash** (`ppo_mean_squash`, `robochess/policies.py`'s `SquashedMultiInputActorCriticPolicy`) is
  a documented deviation from SB3's default `MultiInputActorCriticPolicy`: it tanh's the Gaussian mean before
  building the action distribution, in every codepath that builds one (`forward`, `predict`,
  `get_distribution`, `evaluate_actions`). Adopted per rule 2 above.

## (g) Pre-registered diagnostics (reported, not gates)

- **Truncated-in-S2 tripwire** (at least 10%): fired on squash s1 at **29.0%**, consistent with its
  sets-down-but-never-opens pattern; not fired on any other run (7.3% or less).
- **Saturation** (held steps with P(clipped) > 0.9 on dx, dy or dz): cannot fire for the squash variant by
  construction (a tanh-bounded mean caps P(clipped) at about 0.5), so it is uninformative for the adopted
  variant. The default-variant values are in `section4_summary.json`.
- **Toppled hold at step 150**: 0.0 to 2.0% on all PPO runs, 0.0% SAC.
- **Push-assisted fraction of successes**: 0.0 to 0.69%.
