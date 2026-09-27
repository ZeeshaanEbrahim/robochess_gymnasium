# Section 7 held-aware exemption

The captured piece may cross the XY envelope only with bilateral pad contact and no board/pad/table/pedestal/floor contact. Capturing-piece rules and below-board termination are unchanged. `captured_offboard_exempt` marks prevented captured-piece XY off-board termination.

Commands:
- `nice -n 19 pytest tests/test_t3_capture.py tests/test_t3_scene.py -W error::UserWarning`: 35 passed, exit 0.
- `nice -n 19 python results/t3_capture/section7_diagnosis.py`: exit 0.
- Recorded file identity hash comparison: all match, exit 0.
- `set -o pipefail; nice -n 19 pytest tests/ -W error::UserWarning; echo $?`: 510 passed in 147.86s; exit 0.

## Removal reruns

| Square | Stage 1 | Latch | Total removal leg | Max exempt past slab edge mm | Max exempt past referee envelope mm | Release height mm | Release XY m |
|---|---|---:|---:|---:|---:|---:|---|
| h7 | True | 181 | 205 | 48.278513 | 9.278513 | -0.880212 | [0.34562812401005144, 0.3271519110251929] |
| g8 | True | 193 | 218 | 46.102609 | 7.102609 | -0.886591 | [0.34520162671658416, 0.3279895237431816] |
| h8 | True | 194 | 219 | 65.912072 | 26.912072 | -0.887905 | [0.34522831799758297, 0.3277051571953611] |

## Full capture reruns

| Pair | Success | Latch | Stage-2 length | Total | Max exempt past slab mm | Max exempt past envelope mm | Captured release height mm | Captured release XY m |
|---|---|---:|---:|---:|---:|---:|---:|---|
| g6→h8 | True | 194 | 220 | 414 | 65.912072 | 26.912072 | -0.887905 | [0.34522831799758297, 0.3277051571953611] |
| h1→h8 | False | 194 | 96 | 290 | 65.912072 | 26.912072 | -0.887905 | [0.34522831799758297, 0.3277051571953611] |
| h8→h1 | False | 121 | 170 | 291 | 0.000000 | 0.000000 | -0.585799 | [0.3451391565402639, 0.3207405526036707] |

Lengths and excursion are macro-step measurements; stage-2 length is total minus latch. Slab edge is the physical board edge; the referee envelope adds its existing half-square margin. All new clearances use only geoms with contype or conaffinity nonzero.

## Collision-only clearance

| Episode | Stage | Minimum mm | Pair | Step |
|---|---|---:|---|---:|
| remove h7 | remove | 382.319787 | ur_gripper_left_pad1 ↔ piece_a_head | 4 |
| remove g8 | remove | 349.505227 | ur_gripper_left_pad1 ↔ piece_a_head | 6 |
| remove h8 | remove | 381.765556 | ur_gripper_left_pad1 ↔ piece_a_head | 4 |
| g6→h8 | remove | 104.095080 | ur_gripper_left_pad1 ↔ piece_a_head | 140 |
| g6→h8 | move | 66.566843 | ur_gripper_right_pad1 ↔ piece_b_head | 219 |
| h1→h8 | remove | 52.558060 | ur_gripper_left_pad1 ↔ piece_a_base | 200 |
| h1→h8 | move | 59.864122 | ur_gripper_right_pad1 ↔ piece_b_base | 261 |
| h8→h1 | remove | 389.694117 | ur_gripper_right_pad1 ↔ piece_a_head | 4 |
| h8→h1 | move | 66.812982 | ur_gripper_right_pad1 ↔ piece_b_head | 147 |

## a8 near-contact

Collision-only signed minimum: -0.018232699 mm at step 48; geom_43@ur_gripper_right_follower ↔ piece_a_head.
Parked piece peak displacement: 0.132915374 mm; peak tilt: 0.610415818 degrees.

Contact list at that step (full raw list, including repeated contact points):
```json
[
  {
    "pair": [
      "piece_a_head",
      "geom_43@ur_gripper_right_follower"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_follower"
    ],
    "distance_mm": -0.018232699322972765
  },
  {
    "pair": [
      "piece_a_head",
      "geom_43@ur_gripper_right_follower"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_follower"
    ],
    "distance_mm": -0.018232699322972765
  },
  {
    "pair": [
      "piece_a_head",
      "geom_43@ur_gripper_right_follower"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_follower"
    ],
    "distance_mm": -0.018232699322972765
  },
  {
    "pair": [
      "piece_a_base",
      "board_slab"
    ],
    "bodies": [
      "piece_a",
      "board"
    ],
    "distance_mm": -0.028383975532254302
  },
  {
    "pair": [
      "piece_a_base",
      "board_slab"
    ],
    "bodies": [
      "piece_a",
      "board"
    ],
    "distance_mm": -0.028383975532254302
  },
  {
    "pair": [
      "piece_a_base",
      "board_slab"
    ],
    "bodies": [
      "piece_a",
      "board"
    ],
    "distance_mm": -0.028383975532254302
  },
  {
    "pair": [
      "piece_a_base",
      "board_slab"
    ],
    "bodies": [
      "piece_a",
      "board"
    ],
    "distance_mm": -0.028383975532254302
  },
  {
    "pair": [
      "piece_a_base",
      "board_slab"
    ],
    "bodies": [
      "piece_a",
      "board"
    ],
    "distance_mm": -0.028383975532254302
  },
  {
    "pair": [
      "piece_b_base",
      "board_slab"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.007749620465486086
  },
  {
    "pair": [
      "piece_b_base",
      "board_slab"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.007749620465486086
  },
  {
    "pair": [
      "piece_b_base",
      "board_slab"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.007749620465486086
  },
  {
    "pair": [
      "piece_b_base",
      "board_slab"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.007749620465486086
  },
  {
    "pair": [
      "piece_b_base",
      "board_slab"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.007749620465486086
  }
]
```

This is actual collision contact, not merely overlap of visual geoms. No geometry or expert fix was made.

## Full-capture failures

### h1→h8

Phase move:traverse; reason piece_off_board; off-board piece piece_a; referee contact pairs [].
```json
[
  {
    "pair": [
      "piece_a_stem",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.1975491383620926
  },
  {
    "pair": [
      "piece_a_stem",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.1975491383620926
  },
  {
    "pair": [
      "piece_a_stem",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.1975491383620926
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6472699246245929
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6779562492348068
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.624875249485483
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.5713832354103526
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.6010482127787893
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.5946430459077973
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.0073770470892511805
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.0073770470892511805
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.0073770470892511805
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.0073770470892511805
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.0073770470892511805
  }
]
```

### h8→h1

Phase move:traverse; reason piece_off_board; off-board piece piece_a; referee contact pairs [].
```json
[
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6538561272769472
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6309966223500447
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6317012425138858
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_right_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_right_pad"
    ],
    "distance_mm": -0.6279408318896877
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.5149692930862086
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad1"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.48712301342856457
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.48818744018517235
  },
  {
    "pair": [
      "piece_a_head",
      "ur_gripper_left_pad2"
    ],
    "bodies": [
      "piece_a",
      "ur_gripper_left_pad"
    ],
    "distance_mm": -0.4833879366493241
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.006799474328299241
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.006799474328299241
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.006799474328299241
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.006799474328299241
  },
  {
    "pair": [
      "piece_b_base",
      "capture_pad"
    ],
    "bodies": [
      "piece_b",
      "board"
    ],
    "distance_mm": -0.006799474328299241
  }
]
```

Both remaining failures are capturing-piece joint traverses. The approved captured-piece exemption does not cover piece_a.

## Own test setup fix

The first constructed drag test had no table contact at an exact-surface pose (targeted suite: 1 failed, 34 passed, exit 1). Gravity integration for one existing frame_skip before positioning the closed gripper produces real table contact; the corrected test asserts both finger contacts and table contact, then piece_off_board termination. No production correction beyond the approved rule was needed.
