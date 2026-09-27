# T3 section 4 measurements

Command: `nice -n 19 python scripts/check_t3_expert.py` — exit **0**.
All episodes use `max_episode_steps=1000`, joint absolute mode and seed 0.
The expert source, constants and T3 executable rules were unchanged.

## Measurement definitions

- `Stage-1 step` is the wrapper flag latch. `Stage-2 length` is total minus that latch, including the remaining removal release/retreat.
- Interaction stages follow the controller sequences: **remove** measures gripper versus capturing piece_a; **move** measures gripper versus captured piece_b.
- Signed geom distances include all gripper geoms (visual and collision) and all geoms on the other piece. Measurements are at reset, controller handoff and macro-step boundaries, not every physics substep.
- Peak displacement is the 3D norm relative to that other piece’s position at the start of the controller sequence. Tilt is absolute tilt from upright.
- Pad-edge distances are signed centre-to-edge distances at the last bilateral contact; edge order is left, right, near (low y), far (high y). Negative means outside.
- D27 release records are last-bilateral-contact records. For a failure while still held, they are labelled **unreleased**, not reported as an actual release event.

## Summary

Successes: **17/20**.
Successful total lengths: min **258**, median **281**, max **380**.

| Step cap | Successes fitting |
|---|---:|
| 300 | 13 |
| 400 | 17 |
| 500 | 17 |
| 600 | 17 |

## Pair outcomes

| Pair | Success | Stage-1 step | Stage-2 length | Total |
|---|---|---:|---:|---:|
| c4→f6 | yes | 141 | 160 | 301 |
| b3→e7 | yes | 179 | 160 | 339 |
| e4→e5 | yes | 132 | 149 | 281 |
| d4→c4 | yes | 128 | 145 | 273 |
| e4→f5 | yes | 131 | 150 | 281 |
| c3→b2 | yes | 127 | 146 | 273 |
| g2→h2 | yes | 119 | 139 | 258 |
| a2→h2 | yes | 119 | 159 | 278 |
| f4→h3 | yes | 119 | 149 | 268 |
| b5→h1 | yes | 121 | 170 | 291 |
| g3→g2 | yes | 121 | 150 | 271 |
| h3→c6 | yes | 142 | 152 | 294 |
| h2→a7 | yes | 180 | 157 | 337 |
| b6→a8 | yes | 197 | 183 | 380 |
| c2→a1 | yes | 127 | 153 | 280 |
| g6→h8 | no | — | — | 142 |
| c1→c4 | yes | 128 | 154 | 282 |
| f1→f3 | yes | 123 | 151 | 274 |
| h1→h8 | no | — | — | 142 |
| h8→h1 | no | 121 | 170 | 291 |

## c4→f6

Success: **True**; stage-1 step: **141**; stage-2 length: **160**; total: **301**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 21 | 35 | 9 | 25 | 22 | 18 | 17 | 19 |
| move | 22 | 25 | 9 | 19 | 21 | 25 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.883085  |
| Captured XY world position (mm) | 345.219202, 326.542400 |
| Captured pad-edge distances (mm), left/right/near/far | 33.219202, 44.780798, 40.542400, 37.457600 |
| Capturing height above board (mm), D27 record | -0.504911  |
| Capturing XY error (mm), D27 record | 9.455864 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 200.151898 | 0.012859385 | 0.002288190 | geom_50@ur_gripper_left_coupler ↔ piece_a_base | 36 |
| move | piece_b | 57.284271 | 0.000020064 | 0.002257281 | ur_gripper_right_pad1 ↔ piece_b_head | 167 |

## b3→e7

Success: **True**; stage-1 step: **179**; stage-2 length: **160**; total: **339**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 20 | 34 | 45 | 25 | 24 | 17 | 17 | 19 |
| move | 23 | 25 | 9 | 18 | 24 | 25 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.852280  |
| Captured XY world position (mm) | 345.595019, 330.004183 |
| Captured pad-edge distances (mm), left/right/near/far | 33.595019, 44.404981, 44.004183, 33.995817 |
| Capturing height above board (mm), D27 record | -0.491679  |
| Capturing XY error (mm), D27 record | 12.512785 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 320.362212 | 0.012842066 | 0.002663515 | geom_50@ur_gripper_left_coupler ↔ piece_a_base | 34 |
| move | piece_b | 54.671831 | 0.000022724 | 0.002459920 | ur_gripper_right_pad1 ↔ piece_b_head | 202 |

## e4→e5

Success: **True**; stage-1 step: **132**; stage-2 length: **149**; total: **281**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 22 | 29 | 9 | 20 | 21 | 20 | 17 | 19 |
| move | 21 | 24 | 9 | 19 | 17 | 20 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.887872  |
| Captured XY world position (mm) | 345.064986, 325.162185 |
| Captured pad-edge distances (mm), left/right/near/far | 33.064986, 44.935014, 39.162185, 38.837815 |
| Capturing height above board (mm), D27 record | -0.401020  |
| Capturing XY error (mm), D27 record | 8.804830 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 54.081224 | 0.012788732 | 0.002073744 | ur_gripper_left_pad1 ↔ piece_a_base | 55 |
| move | piece_b | 57.071953 | 0.000019420 | 0.002208889 | ur_gripper_right_pad1 ↔ piece_b_head | 158 |

## d4→c4

Success: **True**; stage-1 step: **128**; stage-2 length: **145**; total: **273**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 22 | 25 | 9 | 19 | 23 | 19 | 17 | 19 |
| move | 22 | 24 | 9 | 19 | 13 | 19 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.586539  |
| Captured XY world position (mm) | 344.872015, 326.401708 |
| Captured pad-edge distances (mm), left/right/near/far | 32.872015, 45.127985, 40.401708, 37.598292 |
| Capturing height above board (mm), D27 record | -0.354350  |
| Capturing XY error (mm), D27 record | 7.545684 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 9.516598 | 0.012788732 | 0.002073744 | geom_42@ur_gripper_right_follower ↔ piece_a_head | 38 |
| move | piece_b | 56.870706 | 0.000008069 | 0.001934845 | ur_gripper_right_pad1 ↔ piece_b_head | 154 |

## e4→f5

Success: **True**; stage-1 step: **131**; stage-2 length: **150**; total: **281**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 22 | 30 | 9 | 21 | 20 | 18 | 17 | 19 |
| move | 21 | 24 | 9 | 19 | 17 | 21 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.886262  |
| Captured XY world position (mm) | 344.968758, 324.519632 |
| Captured pad-edge distances (mm), left/right/near/far | 32.968758, 45.031242, 38.519632, 39.480368 |
| Capturing height above board (mm), D27 record | -0.447571  |
| Capturing XY error (mm), D27 record | 9.052344 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 48.840928 | 0.012788732 | 0.002073744 | ur_gripper_left_pad1 ↔ piece_a_base | 38 |
| move | piece_b | 57.240023 | 0.000005190 | 0.002060559 | ur_gripper_right_pad1 ↔ piece_b_head | 157 |

## c3→b2

Success: **True**; stage-1 step: **127**; stage-2 length: **146**; total: **273**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 25 | 9 | 18 | 23 | 19 | 17 | 19 |
| move | 23 | 24 | 8 | 18 | 14 | 19 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.533775  |
| Captured XY world position (mm) | 345.714025, 326.675825 |
| Captured pad-edge distances (mm), left/right/near/far | 33.714025, 44.285975, 40.675825, 37.324175 |
| Capturing height above board (mm), D27 record | -0.223756  |
| Capturing XY error (mm), D27 record | 4.979696 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 54.864003 | 0.012888092 | 0.002323611 | ur_gripper_left_pad1 ↔ piece_a_head | 77 |
| move | piece_b | 55.252119 | 0.000009503 | 0.002018001 | ur_gripper_right_pad1 ↔ piece_b_head | 154 |

## g2→h2

Success: **True**; stage-1 step: **119**; stage-2 length: **139**; total: **258**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 9 | 18 | 15 | 19 | 17 | 19 |
| move | 17 | 25 | 8 | 18 | 13 | 19 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.577677  |
| Captured XY world position (mm) | 344.812755, 321.831479 |
| Captured pad-edge distances (mm), left/right/near/far | 32.812755, 45.187245, 35.831479, 42.168521 |
| Capturing height above board (mm), D27 record | -0.515282  |
| Capturing XY error (mm), D27 record | 6.745420 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 5.407627 | 0.012963428 | 0.002336717 | geom_54@ur_gripper_left_follower ↔ piece_a_head | 40 |
| move | piece_b | 11.004731 | 0.000007087 | 0.002578051 | geom_42@ur_gripper_right_follower ↔ piece_b_head | 258 |

## a2→h2

Success: **True**; stage-1 step: **119**; stage-2 length: **159**; total: **278**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 9 | 18 | 15 | 19 | 17 | 19 |
| move | 24 | 24 | 12 | 18 | 24 | 18 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.577677  |
| Captured XY world position (mm) | 344.812755, 321.831479 |
| Captured pad-edge distances (mm), left/right/near/far | 32.812755, 45.187245, 35.831479, 42.168521 |
| Capturing height above board (mm), D27 record | -0.513196  |
| Capturing XY error (mm), D27 record | 5.810962 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 385.485520 | 0.012956083 | 0.002642449 | ur_gripper_left_pad1 ↔ piece_a_head | 2 |
| move | piece_b | 10.999189 | 0.000007107 | 0.002578051 | geom_42@ur_gripper_right_follower ↔ piece_b_head | 278 |

## f4→h3

Success: **True**; stage-1 step: **119**; stage-2 length: **149**; total: **268**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 22 | 24 | 9 | 19 | 15 | 19 | 17 | 19 |
| move | 19 | 25 | 9 | 19 | 17 | 21 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.897202  |
| Captured XY world position (mm) | 344.742826, 323.174232 |
| Captured pad-edge distances (mm), left/right/near/far | 32.742826, 45.257174, 37.174232, 40.825768 |
| Capturing height above board (mm), D27 record | -0.876486  |
| Capturing XY error (mm), D27 record | 6.316635 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 101.501591 | 0.012826546 | 0.002344687 | geom_50@ur_gripper_left_coupler ↔ piece_a_head | 38 |
| move | piece_b | 57.640460 | 0.000023364 | 0.002649092 | ur_gripper_right_pad1 ↔ piece_b_base | 268 |

## b5→h1

Success: **True**; stage-1 step: **121**; stage-2 length: **170**; total: **291**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 8 | 18 | 18 | 20 | 17 | 19 |
| move | 23 | 31 | 9 | 22 | 24 | 21 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.585799  |
| Captured XY world position (mm) | 345.139157, 320.740553 |
| Captured pad-edge distances (mm), left/right/near/far | 33.139157, 44.860843, 34.740553, 43.259447 |
| Capturing height above board (mm), D27 record | -0.453055  |
| Capturing XY error (mm), D27 record | 6.266939 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 312.531184 | 0.012856133 | 0.002483512 | ur_gripper_left_pad1 ↔ piece_a_head | 2 |
| move | piece_b | 58.556668 | 0.000009530 | 0.002686227 | ur_gripper_right_pad1 ↔ piece_b_base | 291 |

## g3→g2

Success: **True**; stage-1 step: **121**; stage-2 length: **150**; total: **271**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 25 | 8 | 18 | 17 | 19 | 17 | 19 |
| move | 17 | 25 | 9 | 18 | 15 | 25 | 16 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.581957  |
| Captured XY world position (mm) | 344.685496, 321.371957 |
| Captured pad-edge distances (mm), left/right/near/far | 32.685496, 45.314504, 35.371957, 42.628043 |
| Capturing height above board (mm), D27 record | -0.933945  |
| Capturing XY error (mm), D27 record | 3.869036 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 50.269021 | 0.012853380 | 0.002539151 | ur_gripper_left_pad1 ↔ piece_a_base | 56 |
| move | piece_b | 57.743724 | 0.000016010 | 0.002797456 | ur_gripper_right_pad1 ↔ piece_b_head | 147 |

## h3→c6

Success: **True**; stage-1 step: **142**; stage-2 length: **152**; total: **294**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 21 | 35 | 9 | 25 | 23 | 18 | 17 | 19 |
| move | 15 | 25 | 9 | 19 | 21 | 24 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.887342  |
| Captured XY world position (mm) | 344.949880, 328.395116 |
| Captured pad-edge distances (mm), left/right/near/far | 32.949880, 45.050120, 42.395116, 35.604884 |
| Capturing height above board (mm), D27 record | -0.463188  |
| Capturing XY error (mm), D27 record | 10.219296 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 44.077077 | 0.012928453 | 0.003000253 | ur_gripper_left_pad1 ↔ piece_a_head | 98 |
| move | piece_b | 49.482281 | 0.000031494 | 0.002237455 | ur_gripper_right_pad1 ↔ piece_b_base | 198 |

## h2→a7

Success: **True**; stage-1 step: **180**; stage-2 length: **157**; total: **337**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 20 | 34 | 45 | 25 | 25 | 19 | 17 | 19 |
| move | 15 | 24 | 10 | 18 | 30 | 18 | 18 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.883346  |
| Captured XY world position (mm) | 345.050175, 332.481065 |
| Captured pad-edge distances (mm), left/right/near/far | 33.050175, 44.949825, 46.481065, 31.518935 |
| Capturing height above board (mm), D27 record | 3.684998  |
| Capturing XY error (mm), D27 record | 18.422788 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 2.324982 | 0.012986843 | 0.002633372 | geom_54@ur_gripper_left_follower ↔ piece_a_head | 185 |
| move | piece_b | 5.804268 | 0.000019342 | 0.002537309 | geom_42@ur_gripper_right_follower ↔ piece_b_head | 243 |

## b6→a8

Success: **True**; stage-1 step: **197**; stage-2 length: **183**; total: **380**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 26 | 19 | 17 | 19 |
| move | 23 | 34 | 9 | 25 | 30 | 25 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.858431  |
| Captured XY world position (mm) | 345.002460, 333.651347 |
| Captured pad-edge distances (mm), left/right/near/far | 33.002460, 44.997540, 47.651347, 30.348653 |
| Capturing height above board (mm), D27 record | -0.562429  |
| Capturing XY error (mm), D27 record | 10.931673 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 90.167611 | 0.012853380 | 0.002539151 | ur_gripper_left_pad1 ↔ piece_a_head | 141 |
| move | piece_b | 66.563813 | 0.000024882 | 0.002193121 | ur_gripper_right_pad1 ↔ piece_b_head | 220 |

## c2→a1

Success: **True**; stage-1 step: **127**; stage-2 length: **153**; total: **280**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 8 | 18 | 24 | 20 | 17 | 19 |
| move | 23 | 25 | 8 | 18 | 19 | 20 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.601036  |
| Captured XY world position (mm) | 344.654932, 326.811381 |
| Captured pad-edge distances (mm), left/right/near/far | 32.654932, 45.345068, 40.811381, 37.188619 |
| Capturing height above board (mm), D27 record | -0.188768  |
| Capturing XY error (mm), D27 record | 3.636339 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 60.691968 | 0.012853380 | 0.002539151 | ur_gripper_right_pad1 ↔ piece_a_head | 75 |
| move | piece_b | 55.534009 | 0.000017261 | 0.002011875 | ur_gripper_right_pad1 ↔ piece_b_head | 154 |

## g6→h8

Success: **False**; stage-1 step: **None**; stage-2 length: **None**; total: **142**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 3 | — | — | — |
| move | — | — | — | — | — | — | — | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | 65.591001 (unreleased) |
| Captured XY world position (mm) | 355.328337, 603.009143 |
| Captured pad-edge distances (mm), left/right/near/far | 43.328337, 34.671663, 317.009143, -239.009143 |
| Capturing height above board (mm), D27 record | —  |
| Capturing XY error (mm), D27 record | — |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 104.095080 | 0.012842066 | 0.002663515 | ur_gripper_left_pad1 ↔ piece_a_head | 140 |
| move | not reached | — | — | — | — | — |

### Failure

- Phase: `remove:traverse`; `terminated_reason=piece_off_board`; off-board piece: `piece_b`.
- Referee contact pair: none: off-board boundary termination, not illegal contact.
- Active contact pairs (legal grasp/support contacts):
  - `piece_a_base ↔ board_slab`
  - `piece_b_head ↔ ur_gripper_left_pad1`
  - `piece_b_head ↔ ur_gripper_left_pad2`
  - `piece_b_head ↔ ur_gripper_right_pad1`
  - `piece_b_head ↔ ur_gripper_right_pad2`

## c1→c4

Success: **True**; stage-1 step: **128**; stage-2 length: **154**; total: **282**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 22 | 25 | 9 | 19 | 23 | 19 | 17 | 19 |
| move | 22 | 26 | 8 | 18 | 21 | 20 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.586539  |
| Captured XY world position (mm) | 344.872015, 326.401708 |
| Captured pad-edge distances (mm), left/right/near/far | 32.872015, 45.127985, 40.401708, 37.598292 |
| Capturing height above board (mm), D27 record | -0.338137  |
| Capturing XY error (mm), D27 record | 9.278680 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 204.362275 | 0.012928453 | 0.003000253 | ur_gripper_right_pad1 ↔ piece_a_base | 38 |
| move | piece_b | 56.512378 | 0.000008077 | 0.001934845 | ur_gripper_right_pad1 ↔ piece_b_head | 154 |

## f1→f3

Success: **True**; stage-1 step: **123**; stage-2 length: **151**; total: **274**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 9 | 18 | 19 | 19 | 17 | 19 |
| move | 20 | 26 | 10 | 18 | 19 | 19 | 14 | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.900789  |
| Captured XY world position (mm) | 344.549841, 323.462169 |
| Captured pad-edge distances (mm), left/right/near/far | 32.549841, 45.450159, 37.462169, 40.537831 |
| Capturing height above board (mm), D27 record | -0.295997  |
| Capturing XY error (mm), D27 record | 8.935797 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 129.799886 | 0.012897276 | 0.002500961 | ur_gripper_right_pad1 ↔ piece_a_base | 40 |
| move | piece_b | 57.027699 | 0.000014560 | 0.002575906 | ur_gripper_right_pad1 ↔ piece_b_head | 149 |

## h1→h8

Success: **False**; stage-1 step: **None**; stage-2 length: **None**; total: **142**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 3 | — | — | — |
| move | — | — | — | — | — | — | — | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | 65.591001 (unreleased) |
| Captured XY world position (mm) | 355.328337, 603.009143 |
| Captured pad-edge distances (mm), left/right/near/far | 43.328337, 34.671663, 317.009143, -239.009143 |
| Capturing height above board (mm), D27 record | —  |
| Capturing XY error (mm), D27 record | — |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 340.113031 | 0.013077736 | 0.002593491 | ur_gripper_left_pad1 ↔ piece_a_base | 142 |
| move | not reached | — | — | — | — | — |

### Failure

- Phase: `remove:traverse`; `terminated_reason=piece_off_board`; off-board piece: `piece_b`.
- Referee contact pair: none: off-board boundary termination, not illegal contact.
- Active contact pairs (legal grasp/support contacts):
  - `piece_a_base ↔ board_slab`
  - `piece_b_head ↔ ur_gripper_left_pad1`
  - `piece_b_head ↔ ur_gripper_left_pad2`
  - `piece_b_head ↔ ur_gripper_right_pad1`
  - `piece_b_head ↔ ur_gripper_right_pad2`

## h8→h1

Success: **False**; stage-1 step: **121**; stage-2 length: **170**; total: **291**.

### Phase lengths

| Controller | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 24 | 8 | 18 | 18 | 20 | 17 | 19 |
| move | 25 | 45 | 45 | 25 | 4 | — | — | — |

### Release records

| Measurement | Value |
|---|---|
| Captured height above pad (mm) | -0.585799  |
| Captured XY world position (mm) | 345.139157, 320.740553 |
| Captured pad-edge distances (mm), left/right/near/far | 33.139157, 44.860843, 34.740553, 43.259447 |
| Capturing height above board (mm), D27 record | 86.435020 (unreleased) |
| Capturing XY error (mm), D27 record | 263.753861 |

### Other-piece interactions

| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |
|---|---|---:|---:|---:|---|---:|
| remove | piece_a | 389.694117 | 0.013077736 | 0.002593491 | ur_gripper_right_pad1 ↔ piece_a_head | 4 |
| move | piece_b | 66.812982 | 0.000009530 | 0.002686227 | ur_gripper_right_pad1 ↔ piece_b_head | 147 |

### Failure

- Phase: `move:traverse`; `terminated_reason=piece_off_board`; off-board piece: `piece_a`.
- Referee contact pair: none: off-board boundary termination, not illegal contact.
- Active contact pairs (legal grasp/support contacts):
  - `piece_a_head ↔ ur_gripper_left_pad1`
  - `piece_a_head ↔ ur_gripper_left_pad2`
  - `piece_a_head ↔ ur_gripper_right_pad1`
  - `piece_a_head ↔ ur_gripper_right_pad2`
  - `piece_b_base ↔ capture_pad`

## Diagnosis

- g6→h8 and h1→h8 fail at the same removal traverse step. Captured piece_b is still pinched, at x=0.355328 m and y=0.603009 m: beyond the board’s side envelope and outside the pad footprint. Removal/place/release is never reached.
- h8→h1 completes removal, then capturing piece_a crosses the board boundary during its joint-space traverse while still pinched.
- h1→h8 fails before the capturing-piece traverse. It therefore does not reproduce that original D10 traversal in this episode.
- No fixes were made to the expert, constants, physics or T3 rules.
