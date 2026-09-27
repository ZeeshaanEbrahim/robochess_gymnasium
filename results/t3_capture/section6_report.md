# Section 6 diagnosis

Command: `nice -n 19 python results/t3_capture/section6_diagnosis.py` — exit 0.
Budget 1000; unchanged joint expert; seed 0. Distances measured at macro-step boundaries, including visual geoms. Pad distances are signed piece-centre to edge, not base-radius clearance. Failed held pieces have last-contact records, not actual releases.

## Removal sweep

| Square | Parked | Stage 1 | Latch | Height mm | Nearest edge mm | Minimum gripper clearance mm |
|---|---|---|---:|---:|---:|---:|
| a1 | a8 | yes | 127 | -0.601036 | 32.654932 | 411.254879 |
| b1 | a8 | yes | 128 | -0.596422 | 32.706671 | 424.709703 |
| c1 | a8 | yes | 126 | -0.601982 | 32.502359 | 442.444896 |
| f1 | a8 | yes | 127 | -0.594155 | 32.822750 | 462.893951 |
| g1 | a8 | yes | 127 | -0.569299 | 32.966659 | 456.145785 |
| h1 | a8 | yes | 121 | -0.585799 | 33.139157 | 448.171379 |
| a2 | a8 | yes | 129 | -0.607957 | 32.582496 | 380.079797 |
| b2 | a8 | yes | 127 | -0.533775 | 33.714025 | 391.581942 |
| c2 | a8 | yes | 127 | -0.619220 | 32.536013 | 402.733720 |
| d2 | a8 | yes | 125 | -0.602132 | 32.403065 | 418.550433 |
| e2 | a8 | yes | 124 | -0.597297 | 32.415314 | 433.063984 |
| f2 | a8 | yes | 123 | -0.588195 | 32.667868 | 437.564875 |
| g2 | a8 | yes | 121 | -0.581957 | 32.685496 | 437.750480 |
| h2 | a8 | yes | 119 | -0.577677 | 32.812755 | 437.584549 |
| a3 | a8 | yes | 127 | -0.618560 | 32.734377 | 341.087847 |
| b3 | a8 | yes | 127 | -0.606752 | 32.710492 | 353.160975 |
| c3 | a8 | yes | 124 | -0.610462 | 32.688461 | 364.065743 |
| d3 | a8 | yes | 124 | -0.592104 | 33.086066 | 381.924646 |
| e3 | a8 | yes | 124 | -0.900511 | 32.526730 | 399.951121 |
| f3 | a8 | yes | 123 | -0.900789 | 32.549841 | 413.305653 |
| g3 | a8 | yes | 122 | -0.898326 | 32.650383 | 421.919325 |
| h3 | a8 | yes | 119 | -0.897202 | 32.742826 | 425.562158 |
| a4 | a8 | yes | 133 | -0.892370 | 32.732362 | 281.950854 |
| b4 | a8 | yes | 128 | -0.892922 | 32.698448 | 286.728426 |
| c4 | a8 | yes | 128 | -0.586539 | 32.872015 | 300.906611 |
| d4 | a8 | yes | 127 | -0.580013 | 32.956199 | 330.449714 |
| e4 | a8 | yes | 126 | -0.575190 | 32.915924 | 367.669881 |
| f4 | a8 | yes | 123 | -0.895369 | 32.723709 | 389.488966 |
| g4 | a8 | yes | 123 | -0.891498 | 32.935581 | 404.234801 |
| h4 | a8 | yes | 126 | -0.883910 | 33.158431 | 416.069773 |
| a5 | a8 | yes | 141 | -0.888492 | 32.871601 | 205.732880 |
| b5 | a8 | yes | 139 | -0.584073 | 33.138263 | 208.327569 |
| c5 | a8 | yes | 136 | -0.565482 | 33.142091 | 228.102758 |
| d5 | a8 | yes | 131 | -0.568870 | 32.969447 | 264.821457 |
| e5 | a8 | yes | 132 | -0.887872 | 33.064986 | 316.164403 |
| f5 | a8 | yes | 131 | -0.886262 | 32.968758 | 363.221147 |
| g5 | a8 | yes | 136 | -0.883577 | 33.288099 | 386.266417 |
| h5 | a8 | yes | 139 | -0.880763 | 33.462921 | 401.781211 |
| a6 | a8 | yes | 180 | -0.565815 | 32.977538 | 128.347650 |
| b6 | a8 | yes | 142 | -0.885746 | 32.866873 | 132.184834 |
| c6 | a8 | yes | 142 | -0.887342 | 32.949880 | 159.506496 |
| d6 | a8 | yes | 141 | -0.885320 | 32.998678 | 208.855368 |
| e6 | a8 | yes | 140 | -0.884340 | 32.992554 | 270.074256 |
| f6 | a8 | yes | 141 | -0.883085 | 33.219202 | 336.837137 |
| g6 | a8 | yes | 179 | -0.879656 | 33.458579 | 369.069985 |
| h6 | a8 | yes | 181 | -0.855903 | 33.805882 | 391.506632 |
| a7 | a8 | yes | 180 | -0.883346 | 31.518935 | 51.226081 |
| b7 | a8 | yes | 178 | -0.880472 | 33.025009 | 58.308074 |
| c7 | a8 | yes | 180 | -0.862616 | 33.271898 | 102.032847 |
| d7 | a8 | yes | 179 | -0.857581 | 33.384368 | 167.840204 |
| e7 | a8 | yes | 179 | -0.852280 | 33.595019 | 237.658192 |
| f7 | a8 | yes | 179 | -0.840889 | 33.750806 | 310.556681 |
| g7 | a8 | yes | 179 | -0.844742 | 33.809185 | 354.705878 |
| h7 | a8 | no | — | 72.053107 (unreleased) | -140.060285 | 382.319787 |
| a8 | b8 | yes | 198 | -0.867052 | 30.078113 | -0.018233 |
| b8 | a8 | yes | 198 | -0.866226 | 31.129076 | 2.031603 |
| c8 | a8 | yes | 195 | -0.873277 | 32.784517 | 74.536791 |
| d8 | a8 | yes | 190 | -0.878311 | 33.036201 | 149.424796 |
| e8 | a8 | yes | 192 | -0.883638 | 33.161276 | 224.668130 |
| f8 | a8 | yes | 192 | -0.884362 | 33.179525 | 300.782699 |
| g8 | a8 | no | — | 73.588781 (unreleased) | -111.957233 | 349.505227 |
| h8 | a8 | no | — | 65.591001 (unreleased) | -239.009143 | 381.765556 |

## Grid

P=stage 1 latched; F=failed; X=excluded.

```text
    a b c d e f g h
8   P P P P P P F F
7   P P P P P P P F
6   P P P P P P P P
5   P P P P P P P P
4   P P P P P P P P
3   P P P P P P P P
2   P P P P P P P P
1   P P P X X P P P
```

## Removal failures

### h7

Step 128; phase `remove:traverse`; reason `piece_off_board`; off-board piece `piece_b`.
Carried XY at first observed termination: [0.35260865605923764, 0.504060285271443] m. Referee contact pairs: [].
Active contacts:
- `piece_a_base ↔ board_slab`
- `piece_b_head ↔ ur_gripper_left_pad1`
- `piece_b_head ↔ ur_gripper_left_pad2`
- `piece_b_head ↔ ur_gripper_right_pad1`
- `piece_b_head ↔ ur_gripper_right_pad2`
- `piece_b_stem ↔ ur_gripper_right_pad1`

### g8

Step 144; phase `remove:traverse`; reason `piece_off_board`; off-board piece `piece_b`.
Carried XY at first observed termination: [0.35450535804532163, 0.4759572333777127] m. Referee contact pairs: [].
Active contacts:
- `piece_a_base ↔ board_slab`
- `piece_b_head ↔ ur_gripper_left_pad1`
- `piece_b_head ↔ ur_gripper_left_pad2`
- `piece_b_head ↔ ur_gripper_right_pad1`
- `piece_b_head ↔ ur_gripper_right_pad2`

### h8

Step 142; phase `remove:traverse`; reason `piece_off_board`; off-board piece `piece_b`.
Carried XY at first observed termination: [0.35532833677229086, 0.6030091425484359] m. Referee contact pairs: [].
Active contacts:
- `piece_a_base ↔ board_slab`
- `piece_b_head ↔ ur_gripper_left_pad1`
- `piece_b_head ↔ ur_gripper_left_pad2`
- `piece_b_head ↔ ur_gripper_right_pad1`
- `piece_b_head ↔ ur_gripper_right_pad2`

## Extreme full captures

| Pair | Success | Latch | Stage-2 length | Total |
|---|---|---:|---:|---:|
| h2→a8 | True | 197 | 160 | 357 |
| g1→b8 | True | 198 | 161 | 359 |
| a1→a8 | True | 197 | 168 | 365 |
| h4→a7 | True | 180 | 166 | 346 |
| f1→a8 | True | 197 | 166 | 363 |

### h2→a8

| Stage | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 26 | 19 | 17 | 19 |
| move | 15 | 24 | 10 | 18 | 30 | 15 | 25 | — |

Captured release height: -0.858431 mm; XY: [0.3450024595615417, 0.33365134670349994] m.
Pad edges (left/right/near/far): 33.002460 / 44.997540 / 47.651347 / 30.348653 mm.
Capturing release height: 5.706966 mm; XY error: 21.067740 mm.

| Stage | Min clearance mm | Peak displacement mm | Peak tilt degrees | Closest pair | Step |
|---|---:|---:|---:|---|---:|
| remove | 2.299142 | 0.012986843 | 0.002633372 | geom_54@ur_gripper_left_follower ↔ piece_a_head | 201 |
| move | 6.395576 | 0.000024811 | 0.002193121 | geom_42@ur_gripper_right_follower ↔ piece_b_head | 259 |

Termination: `success`. No failure; order event None; removal-lost event None.

### g1→b8

| Stage | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 23 | 45 | 45 | 25 | 26 | 20 | 17 | 19 |
| move | 19 | 25 | 8 | 18 | 30 | 25 | 14 | — |

Captured release height: -0.866226 mm; XY: [0.3451285760715096, 0.3328709241375136] m.
Pad edges (left/right/near/far): 33.128576 / 44.871424 / 46.870924 / 31.129076 mm.
Capturing release height: -0.551224 mm; XY error: 16.072212 mm.

| Stage | Min clearance mm | Peak displacement mm | Peak tilt degrees | Closest pair | Step |
|---|---:|---:|---:|---|---:|
| remove | 94.845267 | 0.012956083 | 0.002642449 | geom_50@ur_gripper_left_coupler ↔ piece_a_head | 195 |
| move | 60.485840 | 0.000024793 | 0.002306157 | ur_gripper_right_pad1 ↔ piece_b_head | 221 |

Termination: `success`. No failure; order event None; removal-lost event None.

### a1→a8

| Stage | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 26 | 19 | 17 | 19 |
| move | 25 | 25 | 8 | 18 | 30 | 25 | 14 | — |

Captured release height: -0.858431 mm; XY: [0.3450024595615417, 0.33365134670349994] m.
Pad edges (left/right/near/far): 33.002460 / 44.997540 / 47.651347 / 30.348653 mm.
Capturing release height: -0.555038 mm; XY error: 14.027945 mm.

| Stage | Min clearance mm | Peak displacement mm | Peak tilt degrees | Closest pair | Step |
|---|---:|---:|---:|---|---:|
| remove | 441.570260 | 0.013077736 | 0.002593524 | ur_gripper_left_pad1 ↔ piece_a_head | 0 |
| move | 55.206935 | 0.000024833 | 0.002193121 | ur_gripper_right_pad1 ↔ piece_b_head | 221 |

Termination: `success`. No failure; order event None; removal-lost event None.

### h4→a7

| Stage | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 20 | 34 | 45 | 25 | 25 | 19 | 17 | 19 |
| move | 17 | 27 | 9 | 20 | 30 | 25 | 14 | — |

Captured release height: -0.883346 mm; XY: [0.34505017473243177, 0.33248106481727085] m.
Pad edges (left/right/near/far): 33.050175 / 44.949825 / 46.481065 / 31.518935 mm.
Capturing release height: -0.524580 mm; XY error: 9.569163 mm.

| Stage | Min clearance mm | Peak displacement mm | Peak tilt degrees | Closest pair | Step |
|---|---:|---:|---:|---|---:|
| remove | 45.043081 | 0.012886473 | 0.002369397 | ur_gripper_right_pad1 ↔ piece_a_head | 131 |
| move | 63.827339 | 0.000019347 | 0.002537309 | ur_gripper_right_pad1 ↔ piece_b_head | 205 |

Termination: `success`. No failure; order event None; removal-lost event None.

### f1→a8

| Stage | approach | descend | close | lift | traverse | place | release | retreat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| remove | 24 | 45 | 45 | 25 | 26 | 19 | 17 | 19 |
| move | 20 | 26 | 10 | 18 | 30 | 24 | 15 | — |

Captured release height: -0.858431 mm; XY: [0.3450024595615417, 0.33365134670349994] m.
Pad edges (left/right/near/far): 33.002460 / 44.997540 / 47.651347 / 30.348653 mm.
Capturing release height: -0.589788 mm; XY error: 16.562449 mm.

| Stage | Min clearance mm | Peak displacement mm | Peak tilt degrees | Closest pair | Step |
|---|---:|---:|---:|---|---:|
| remove | 160.756785 | 0.012897276 | 0.002500961 | geom_50@ur_gripper_left_coupler ↔ piece_a_base | 195 |
| move | 58.264723 | 0.000024833 | 0.002193122 | ur_gripper_right_pad1 ↔ piece_b_head | 221 |

Termination: `success`. No failure; order event None; removal-lost event None.

## Summary

{
  "removable": 59,
  "removal_episodes": 62,
  "failing_squares": [
    "h7",
    "g8",
    "h8"
  ],
  "extreme_successes": 5,
  "combined_successes": 22,
  "combined_episodes": 25,
  "longest_pair": "b6\u2192a8",
  "longest_length": 380,
  "fit_budgets": {
    "400": 22,
    "500": 22,
    "600": 22
  }
}

All removal failures are carried-piece boundary exits during joint traverse; no illegal referee contact pair was recorded. h8 exactly repeats the section 4 termination XY and step. This is consistent with the D10 joint-path limitation; no expert or physics change was tested.

The a8 sweep has minimum signed clearance -0.018232699 mm at step 48 for geom_42@ur_gripper_right_follower ↔ piece_a_head. This measurement includes visual geoms; it alone does not establish a collision contact. Other-piece peak displacement 0.132915374 mm; tilt 0.610415818 degrees.
