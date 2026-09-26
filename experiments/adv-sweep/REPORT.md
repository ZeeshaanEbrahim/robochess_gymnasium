# Adversarial referee sweep: section 3 follow-up and section 4

Simulation source baseline: a37c8ab. All simulations used the activated ~/venvs/robochess environment, with bytecode writes disabled. Scripts and working outputs remained in /tmp/adv_sweep. This report-only commit is the explicit exception authorized in follow-up item 1. No simulator, referee, harness, config, or test source changed.

Open pushes and the e4→e5 near-target topple are NOT REALISED; attempts are stopped. Push/drop intentions remain OPEN. Section 5 has not run.

## Reward ledger: every reference, drop, and push

D20 is the unstaged T1 shaped return. D21 is the pick-up return scored on the same physical trajectory with pickup_obs enabled. The placement referee can still end these T1 episodes; D21 pickup_success itself does not terminate. No row below had a failure penalty. Means are raw episode return divided by the measured episode length.

| Scenario | Pair | Length | F2 success step | D20 return | D21 return | D20 mean/step | D21 mean/step |
|---|---|---:|---:|---:|---:|---:|---:|
| v3 reference (clean=False) | e4→e5 | 146 | 146 | -13.931027 | 254.025222 | -0.095418 | 1.739899 |
| v3 reference (clean=False) | b7→g2 | 193 | 193 | -69.746659 | 488.648079 | -0.361382 | 2.531855 |
| v3 reference (clean=True) | c1→f6 | 207 | 207 | -65.417887 | 496.245735 | -0.316028 | 2.397322 |
| v3 reference (clean=True) | f1→c6 | 230 | 230 | -77.430477 | 564.323874 | -0.336654 | 2.453582 |
| drop_10mm | b7→g2 | 178 | 178 | -69.476738 | 462.995277 | -0.390319 | 2.601097 |
| drop_10mm | e4→e5 | 138 | 138 | -13.860456 | 245.896877 | -0.100438 | 1.781861 |
| drop_20mm | b7→g2 | 178 | 178 | -69.631600 | 470.593052 | -0.391189 | 2.643781 |
| drop_20mm | e4→e5 | 138 | 138 | -14.071450 | 261.275616 | -0.101967 | 1.893302 |
| drop_30mm | b7→g2 | 176 | 176 | -69.725003 | 474.391247 | -0.396165 | 2.695405 |
| drop_30mm | e4→e5 | 137 | 137 | -14.279918 | 277.655396 | -0.104233 | 2.026682 |
| drop_5mm | b7→g2 | 184 | 184 | -69.550040 | 470.873765 | -0.377989 | 2.559097 |
| drop_5mm | e4→e5 | 139 | 139 | -13.837223 | 236.725416 | -0.099548 | 1.703061 |
| drop_80mm | b7→g2 | 165 | 165 | -69.894373 | 436.047485 | -0.423602 | 2.642712 |
| drop_80mm | e4→e5 | 122 | 122 | -14.402144 | 223.078018 | -0.118050 | 1.828508 |
| fu_drop_10mm | c1→f6 | 188 | 188 | -65.271804 | 471.199009 | -0.347190 | 2.506378 |
| fu_drop_30mm | c1→f6 | 187 | 187 | -65.729659 | 503.289766 | -0.351496 | 2.691389 |
| fu_drop_80mm | c1→f6 | 173 | 173 | -65.962015 | 446.687403 | -0.381283 | 2.582008 |
| push_closed | b7→g2 | 300 | — | -186.451310 | 99.166442 | -0.621504 | 0.330555 |
| push_closed_base | b7→g2 | 300 | — | -173.558298 | 183.313014 | -0.578528 | 0.611043 |
| push_closed_base | e4→e5 | 300 | — | -47.272825 | 184.716612 | -0.157576 | 0.615722 |
| closed push then open (original) | e4→e5 | 150 | 150 | -16.116454 | 82.712246 | -0.107443 | 0.551415 |
| push_closed_final | b7→g2 | 1000 | — | -213.818750 | 599.914323 | -0.213819 | 0.599914 |
| push_closed_final | e4→e5 | 1000 | — | -71.093669 | 498.661684 | -0.071094 | 0.498662 |
| push_open | b7→g2 | 300 | — | -177.340863 | 96.470291 | -0.591136 | 0.321568 |
| push_open_base | b7→g2 | 300 | — | -176.288485 | 141.096356 | -0.587628 | 0.470321 |
| push_open_base | e4→e5 | 300 | — | -33.653268 | 148.521264 | -0.112178 | 0.495071 |
| push_open | e4→e5 | 300 | — | -30.490812 | 141.498184 | -0.101636 | 0.471661 |
| push_open_final | b7→g2 | 1000 | — | -575.609074 | 512.546528 | -0.575609 | 0.512547 |
| push_open_final | e4→e5 | 1000 | — | -100.194356 | 543.070668 | -0.100194 | 0.543071 |
| push_open_fixed_floor | b7→g2 | 1000 | — | -625.047437 | 248.203985 | -0.625047 | 0.248204 |
| push_open_fixed_floor | e4→e5 | 1000 | — | -147.721669 | 473.875392 | -0.147722 | 0.473875 |
| fu_push_then_open | b7→g2 | 515 | 515 | -183.017225 | 364.971121 | -0.355373 | 0.708682 |
| fu_push_then_open | e4→e5 | 150 | 150 | -16.116454 | 82.712246 | -0.107443 | 0.551415 |

## Six earlier drops that out-paid the non-clean reference

Relabelled **RM, old reward, mechanism unconfirmed**. Those old references are not valid clean references. Five of the six matched-length differences are negative: avoiding the measured negative tail is sufficient to explain their total advantage. For b7→g2 at 5 mm, the tail contributes 0.179470 and the matched-length trajectory contributes another 0.017149. The following is an exact arithmetic decomposition using measured trajectories: total gain = matched-length return difference + omitted reference-tail cost. It confirms a duration contribution, without claiming that changing only the termination time would produce an identical physical counterfactual.

| Pair / drop mm | Drop/reference length | Earlier by steps | D20 gain | Omitted negative reference tail | Matched-length difference |
|---|---:|---:|---:|---:|---:|
| e4→e5, 5 | 139/146 | 7 | 0.093804 | 0.137384 | -0.043580 |
| e4→e5, 10 | 138/146 | 8 | 0.070571 | 0.156974 | -0.086403 |
| b7→g2, 5 | 184/193 | 9 | 0.196619 | 0.179470 | 0.017149 |
| b7→g2, 10 | 178/193 | 15 | 0.269921 | 0.305303 | -0.035382 |
| b7→g2, 20 | 178/193 | 15 | 0.115059 | 0.305303 | -0.190244 |
| b7→g2, 30 | 176/193 | 17 | 0.021656 | 0.346854 | -0.325198 |

## Verified clean references and new c1→f6 drops

| Pair / case | v3 clean | Height before opening mm | Open command step | qpos < 0.2 step / height mm | Last pad contact step / height mm | Final tilt deg | Final distance mm |
|---|---|---:|---:|---|---|---:|---:|
| c1→f6 reference | True | -0.200 | 194 | 207 / -0.006830 | 198 / -0.083780 | 0.001946 | 1.083 |
| f1→c6 reference | True | -0.251 | 217 | 230 / -0.006830 | 221 / -0.091171 | 0.002136 | 1.806 |
| c1→f6 fu_drop_10mm | — | 10.612 | 184 | 187 / -0.006353 | 183 / 10.611976 | 0.002232 | 0.889 |
| c1→f6 fu_drop_30mm | — | 30.163 | 182 | 185 / 0.001932 | 181 / 30.163372 | 0.001403 | 0.688 |
| c1→f6 fu_drop_80mm | — | 79.380 | 166 | 169 / -0.383472 | 165 / 79.380060 | 0.002185 | 7.924 |

**Release-height definition correction:** all section 3 drop heights were measured at the final closed macro-step, immediately BEFORE the first opening command. They were neither the qpos-threshold crossing height nor a measured last-pad-contact height. The richer follow-up separately records those events at post-step macro boundaries. The last pad-contact boundary can precede the opening command; actual separation inside the following macro-step is not resolved by this boundary measurement. The contact-speed sampler does sample every physics substep.

## Push-then-open, n = 2

| Pair | Push starts | Opening command | F2 success | Max lift mm | Pinched ever | Held ever | grasp_success ever |
|---|---:|---:|---:|---:|---|---|---|
| e4→e5 | 97 | 147 | 150 | 3.547156 | False | False | False |
| b7→g2 | 102 | 512 | 515 | 1.612449 | False | False | False |

Seven returned `_success_terms()` entries, in code order:

| Pair | on_square | resting | released | settled | XY distance m | still | upright |
|---|---|---|---|---|---:|---|---|
| e4→e5 | True | True | True | True | 0.015137729 | True | True |
| b7→g2 | True | True | True | True | 0.028575871 | True | True |

e4→e5 at success: qpos=0.180147784 rad, base height=-0.006889820 mm, angular speed=0.001241578 rad/s, linear speed=0.000000607 m/s, tilt=0.001767061 deg.


b7→g2 at success: qpos=0.180160076 rad, base height=-0.006949425 mm, angular speed=0.000258778 rad/s, linear speed=0.000001001 m/s, tilt=0.001541076 deg.


Code/definition detail: only angular-speed, linear-speed and upright predicates have three-step streaks. on_square, resting and released are immediate tests. In both measured push-then-open cases, F2 success occurred on the very first macro-step with qpos < 0.2, not after three released steps. This is reported without reclassifying the OPEN push intention.

## Section 2 presence-trigger identities and depths

The 118 replayed furniture/self-contact terminations reproduced their reasons. All exact first-trigger pairs are neither finger-finger nor pad-piece. Penetration below is max(0, -MuJoCo contact.dist), measured after the same post-step mj_forward used by the referee. There are 30 distinct first-trigger geom pairs. Raw evidence also retains every presence candidate on each terminal boundary.

| Class | Exact geom pair | Episodes | Min penetration mm | Max penetration mm |
|---|---|---:|---:|---:|
| furniture | <ur_forearm_link>#geom19 ↔ table#geom1 | 11 | 0.453895 | 17.873526 |
| furniture | <ur_forearm_link>#geom20 ↔ table#geom1 | 18 | 1.169443 | 40.836292 |
| furniture | <ur_gripper_base>#geom35 ↔ table#geom1 | 1 | 4.955783 | 4.955783 |
| furniture | <ur_gripper_left_coupler>#geom51 ↔ pedestal#geom2 | 1 | 5.404721 | 5.404721 |
| furniture | <ur_gripper_left_coupler>#geom51 ↔ table#geom1 | 8 | 0.971946 | 19.078854 |
| furniture | <ur_gripper_left_driver>#geom49 ↔ table#geom1 | 2 | 14.752289 | 22.041065 |
| furniture | <ur_gripper_left_follower>#geom55 ↔ pedestal#geom2 | 1 | 0.116716 | 0.116716 |
| furniture | <ur_gripper_left_follower>#geom55 ↔ table#geom1 | 2 | 1.586439 | 1.776803 |
| furniture | <ur_gripper_right_coupler>#geom39 ↔ table#geom1 | 4 | 2.415005 | 15.224532 |
| furniture | <ur_gripper_right_driver>#geom37 ↔ table#geom1 | 4 | 0.946245 | 22.164414 |
| furniture | <ur_upper_arm_link>#geom14 ↔ table#geom1 | 1 | 4.266465 | 4.266465 |
| furniture | <ur_wrist_1_link>#geom24 ↔ table#geom1 | 1 | 1.710724 | 1.710724 |
| furniture | <ur_wrist_2_link>#geom28 ↔ table#geom1 | 32 | 2.136081 | 34.408268 |
| furniture | <ur_wrist_2_link>#geom29 ↔ table#geom1 | 1 | 0.334000 | 0.334000 |
| furniture | pedestal#geom2 ↔ ur_gripper_left_pad1#geom56 | 1 | 0.854749 | 0.854749 |
| furniture | table#geom1 ↔ ur_gripper_left_pad1#geom56 | 4 | 0.067744 | 3.956898 |
| furniture | table#geom1 ↔ ur_gripper_right_pad1#geom44 | 9 | 0.031072 | 1.651400 |
| self | <ur_forearm_link>#geom20 ↔ <ur_gripper_base>#geom35 | 1 | 10.152526 | 10.152526 |
| self | <ur_forearm_link>#geom20 ↔ <ur_gripper_base_mount>#geom33 | 1 | 2.563409 | 2.563409 |
| self | <ur_forearm_link>#geom20 ↔ <ur_gripper_left_coupler>#geom51 | 2 | 5.490772 | 11.324109 |
| self | <ur_forearm_link>#geom20 ↔ <ur_gripper_right_driver>#geom37 | 1 | 10.844398 | 10.844398 |
| self | <ur_forearm_link>#geom20 ↔ <ur_wrist_3_link>#geom31 | 1 | 4.241685 | 4.241685 |
| self | <ur_gripper_base>#geom35 ↔ <ur_upper_arm_link>#geom13 | 1 | 1.638824 | 1.638824 |
| self | <ur_gripper_left_driver>#geom49 ↔ <ur_wrist_1_link>#geom24 | 1 | 1.288975 | 1.288975 |
| self | <ur_gripper_right_coupler>#geom39 ↔ <ur_shoulder_link>#geom8 | 1 | 0.438651 | 0.438651 |
| self | <ur_gripper_right_coupler>#geom39 ↔ <ur_upper_arm_link>#geom13 | 4 | 0.440721 | 1.633156 |
| self | <ur_gripper_right_follower>#geom43 ↔ <ur_shoulder_link>#geom8 | 1 | 0.452327 | 0.452327 |
| self | <ur_shoulder_link>#geom8 ↔ ur_gripper_left_pad1#geom56 | 1 | 0.003196 | 0.003196 |
| self | <ur_upper_arm_link>#geom13 ↔ ur_gripper_right_pad1#geom44 | 1 | 0.010268 | 0.010268 |
| self | <ur_upper_arm_link>#geom14 ↔ <ur_wrist_2_link>#geom28 | 1 | 0.831031 | 0.831031 |

## T2 reward measurement

`success_obstacles_disturbed` is a value of `info["terminated_reason"]`, not an info key. The fixture physically disturbed an obstacle by 10 mm, latched it, restored it, and placed the piece at the target. All measured final outcomes terminated with that reason and info["is_success"]=False.

| Wrapper | Actual step reward | compute_reward at same goals |
|---|---:|---:|
| sparse | 0.000000000 | 1.000000000 |
| shaped | -0.196597716 | -0.000007451 |
| staged | -0.196590266 | -0.000007451 |

Sparse real step does not pay success; sparse HER compute_reward returns 1 and ignores obstacle history and the supplied info. Default/staged shaped rewards do not read the disturbance latch; their compute_reward delegates to the base goal-distance reward. pickup_obs explicitly rejects T2, so the D21 + pickup_obs stack is not supported there. No source fix was made.

## Section 4: clumsy contacts

### Board-force ramp

Absolute action mode was used because the Cartesian workspace floor prevents open pads from reaching the board. The unmodified expert IK supplied absolute joint commands. Downward increments reuse v3 BRAKE_MAX_M and BRAKE_MIN_M; no new force threshold was chosen. Force is the sum of absolute robot-board normal contact forces at each referee macro-step boundary.

Termination: step **1126**, combined robot-board force **251.517260582 N**, reason **bad_contact**. No step at or below 250 N was bad_contact. Piece-board forces are excluded. The lossless per-step force ledger is [board-force-steps.tsv](board-force-steps.tsv).

| Step | Commanded pinch z m | Measured pinch z m | Combined robot-board N | bad_contact |
|---|---:|---:|---:|---|
| 1117 | 0.716500 | 0.747020 | 236.178176050 | False |
| 1118 | 0.716500 | 0.747022 | 236.009717676 | False |
| 1119 | 0.716500 | 0.747023 | 235.842152312 | False |
| 1120 | 0.716500 | 0.747024 | 235.675475558 | False |
| 1121 | 0.716500 | 0.747025 | 235.509683030 | False |
| 1122 | 0.716500 | 0.747027 | 235.344770356 | False |
| 1123 | 0.716500 | 0.747028 | 235.180733183 | False |
| 1124 | 0.716500 | 0.747029 | 235.017567168 | False |
| 1125 | 0.716500 | 0.747030 | 234.855267988 | False |
| 1126 | 0.715000 | 0.747007 | 251.517260582 | True |

### Five open/close cycles

Measured setup: step 120, base 29.372657 mm above rest (nominal 30 mm; existing fine tolerance reused); both pads in contact, piece clear of board. First loss of held state: step 121. First piece-board contact: step 122. No termination or truncation during the cycles.

| Cycle | Open steps | Close steps | Finger self-contact steps |
|---|---|---|---|
| 1 | 121–134 | 135–146 | 144, 145, 146 |
| 2 | 147–161 | 162–173 | 171, 172, 173 |
| 3 | 174–188 | 189–200 | 198, 199, 200 |
| 4 | 201–215 | 216–227 | 225, 226, 227 |
| 5 | 228–242 | 243–254 | 252, 253, 254 |

Section 4 matches the pre-set verdicts: board contact below the ceiling is allowed, board force above it terminates, and left/right finger self-contact does not terminate.

## Full original section 3 tables (labels corrected)

# Section 3 — adversarial probes

All outputs are measured. Pushes and drops retain intended verdict OPEN. No environment or harness source was changed.

## v3 comparison baselines

| Pair | F2 success | v3 clean_place | Release height mm | Shaped return | D21 return |
|---|---|---|---:|---:|---:|
| e4→e5 | True | False | 6.205 | -13.931027 | 254.025222 |
| b7→g2 | True | False | 7.312 | -69.746659 | 488.648079 |

Neither baseline is verified clean under v3. Raw rewards below are compared against those actual v3 runs, not a verified clean comparator.

## Drops

| Pair | Nominal mm | Actual release mm | Success | Final tilt deg | Upright | Final XY distance mm | Peak contact speed m/s | Settle steps after release | Shaped return | D21 return | Shaped greater than v3 reference |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| e4→e5 | 5 | 3.877 | True | 0.001718 | True | 8.700 | 0.126851 | 6 | -13.837223 | 236.725416 | RM, old reward, mechanism unconfirmed; comparator not clean |
| e4→e5 | 10 | 10.637 | True | 0.001600 | True | 5.681 | 0.429673 | 6 | -13.860456 | 245.896877 | RM, old reward, mechanism unconfirmed; comparator not clean |
| e4→e5 | 20 | 20.211 | True | 0.001778 | True | 0.666 | 0.628594 | 6 | -14.071450 | 261.275616 | No |
| e4→e5 | 30 | 29.992 | True | 0.001212 | True | 0.669 | 0.767309 | 6 | -14.279918 | 277.655396 | No |
| e4→e5 | 80 | 80.104 | True | 0.000694 | True | 8.002 | 1.250046 | 7 | -14.402144 | 223.078018 | No |
| b7→g2 | 5 | 5.917 | True | 0.001626 | True | 4.977 | 0.311224 | 6 | -69.550040 | 470.873765 | RM, old reward, mechanism unconfirmed; comparator not clean |
| b7→g2 | 10 | 11.598 | True | 0.016344 | True | 1.055 | 0.469901 | 5 | -69.476738 | 462.995277 | RM, old reward, mechanism unconfirmed; comparator not clean |
| b7→g2 | 20 | 22.415 | True | 0.001707 | True | 2.190 | 0.669798 | 6 | -69.631600 | 470.593052 | RM, old reward, mechanism unconfirmed; comparator not clean |
| b7→g2 | 30 | 32.489 | True | 0.001920 | True | 1.734 | 0.807505 | 6 | -69.725003 | 474.391247 | RM, old reward, mechanism unconfirmed; comparator not clean |
| b7→g2 | 80 | 81.046 | True | 0.001804 | True | 6.256 | 1.251715 | 7 | -69.894373 | 436.047485 | No |

The height controller reused the v3 fine positional tolerance. Actual release heights, including misses of nominal height, are reported above. Contact speed is sampled at every physics substep with piece-board contact. Splitting integration for measurement reproduced the e4 baseline steps and both reward sums exactly.

## All physical push and topple attempts

| Scenario | Pair | Steps | Success | Reason | Final tilt deg | Final XY distance mm | Shaped return | D21 return | Evidence |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| push_closed | b7→g2 | 300 | False | None | 0.002137 | 501.467 | -186.451310 | 99.166442 | `s3_push_closed_b7_g2.json` (evidence.json) |
| push_closed_base | b7→g2 | 300 | False | None | 0.002051 | 539.566 | -173.558298 | 183.313014 | `s3_push_closed_base_b7_g2.json` (evidence.json) |
| push_closed_base | e4→e5 | 300 | False | None | 0.002339 | 234.249 | -47.272825 | 184.716612 | `s3_push_closed_base_e4_e5.json` (evidence.json) |
| push_closed | e4→e5 | 150 | True | success | 0.001767 | 15.138 | -16.116454 | 82.712246 | `s3_push_closed_e4_e5.json` (evidence.json) |
| push_closed_final | b7→g2 | 1000 | False | None | 0.001891 | 28.576 | -213.818750 | 599.914323 | `s3_push_closed_final_b7_g2.json` (evidence.json) |
| push_closed_final | e4→e5 | 1000 | False | None | 0.001046 | 28.854 | -71.093669 | 498.661684 | `s3_push_closed_final_e4_e5.json` (evidence.json) |
| push_open | b7→g2 | 300 | False | None | 103.984456 | 487.842 | -177.340863 | 96.470291 | `s3_push_open_b7_g2.json` (evidence.json) |
| push_open_base | b7→g2 | 300 | False | None | 0.001446 | 551.544 | -176.288485 | 141.096356 | `s3_push_open_base_b7_g2.json` (evidence.json) |
| push_open_base | e4→e5 | 300 | False | None | 0.001603 | 78.000 | -33.653268 | 148.521264 | `s3_push_open_base_e4_e5.json` (evidence.json) |
| push_open | e4→e5 | 300 | False | None | 103.964727 | 52.780 | -30.490812 | 141.498184 | `s3_push_open_e4_e5.json` (evidence.json) |
| push_open_final | b7→g2 | 1000 | False | None | 0.001701 | 550.723 | -575.609074 | 512.546528 | `s3_push_open_final_b7_g2.json` (evidence.json) |
| push_open_final | e4→e5 | 1000 | False | None | 0.001681 | 77.276 | -100.194356 | 543.070668 | `s3_push_open_final_e4_e5.json` (evidence.json) |
| push_open_fixed_floor | b7→g2 | 1000 | False | None | 103.964445 | 495.791 | -625.047437 | 248.203985 | `s3_push_open_fixed_floor_b7_g2.json` (evidence.json) |
| push_open_fixed_floor | e4→e5 | 1000 | False | None | 90.112450 | 108.174 | -147.721669 | 473.875392 | `s3_push_open_fixed_floor_e4_e5.json` (evidence.json) |
| topple | b7→g2 | 204 | True | success | 0.002469 | 26.729 | -70.526355 | 514.582385 | `s3_topple_b7_g2.json` (evidence.json) |
| topple | e4→e5 | 161 | True | success | 0.017602 | 27.651 | -14.866637 | 278.362879 | `s3_topple_e4_e5.json` (evidence.json) |
| topple_e4_near_target | e4→e5 | 152 | True | success | 0.002076 | 24.024 | -16.746214 | 280.362633 | `s3_topple_e4_near_target_e4_e5.json` (evidence.json) |
| topple_final | b7→g2 | 300 | False | None | 103.965152 | 8.615 | -77.252459 | 521.712116 | `s3_topple_final_b7_g2.json` (evidence.json) |
| topple_final | e4→e5 | 300 | False | None | 0.001598 | 40.985 | -25.580293 | 351.484137 | `s3_topple_final_e4_e5.json` (evidence.json) |
| topple_head_bump | b7→g2 | 300 | False | None | 103.965229 | 39.201 | -77.754159 | 527.590842 | `s3_topple_head_bump_b7_g2.json` (evidence.json) |
| topple_head_bump | e4→e5 | 300 | False | None | 0.002136 | 39.711 | -25.106870 | 356.007289 | `s3_topple_head_bump_e4_e5.json` (evidence.json) |
| topple_left_finger | e4→e5 | 300 | False | None | 103.964573 | 72.184 | -29.035095 | 344.632991 | `s3_topple_left_finger_e4_e5.json` (evidence.json) |

## Verification limits

- Closed final pushes reached both target tolerances but stayed unreleased and did not succeed.
- The original e4 closed push followed by opening succeeded without any grasp_success flag; that is a separate observed no-grasp sequence, not a fully open push.
- Fully open push attempts were executed, but no upright on-target open push was realized. They cannot validate the no-grasp success gap.
- The b7→g2 final topple is a valid near-target topple: final tilt above 90 degrees, distance below 30 mm, no success.
- No e4→e5 attempt left a toppled piece within 30 mm. That case remains unverified; transient tilt followed by recovery is not a false success.
- Push attempts used recorded 300- or 1000-step budgets; raw episode returns are not duration-normalized.
- D21 can pay more cumulative reward on a long unsuccessful push or on a lift followed by a drop than on the v3 placement reference. Its target is pickup/holding, not placement; reward totals and budgets are retained without treating this as a new placement reward finding.
- No drop limit is recommended.


## Evidence and execution limits

evidence.json contains original and follow-up per-case summaries, per-step reward triples, follow-up event measurements, contact-depth records, T2 reward measurements, and section 4 measurements keyed by their original filenames. Open-push and e4 topple attempts are stopped and NOT REALISED. No push/drop classification was changed to OK or FS. The obsolete non-clean references are retained for audit, not promoted to clean references. No section 5 or 6 run was performed.
