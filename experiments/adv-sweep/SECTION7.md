# Section 7 — final adversarial referee sweep summary

Branch: `experiments/adv-sweep`. Branch base: **8f122a3cda6f2322613b5948c6626cd793562250** (parent of the first report commit a8517ba; also the current merge base with master). Simulation source baseline: **a37c8ab**. CI commit f4581ae is on this experiment branch; master has a separate CI commit, 524105a.

This report and evidence are the explicitly authorized exception to the read-only run. No source, config, dependency, or test changes were made by this reporting pass. No push was run. REPORT.md preserves the earlier section 3/4 report; its statements that sections 5/6 had not run were historical and are superseded here.

## Verdicts

OK means the measured referee response matches the preset expectation. FT records an unexpected termination of a control intended to be legal; the −Y contact itself is real and the referee correctly applies its contact rule. FS requires observed improper success. RM rows retain known reward-path observations without deciding the intentions of push/drop. OPEN means intent or verification remains unresolved. Not realised means the requested physical configuration was not achieved.

Only the angular-speed, linear-speed and upright F2 predicates have 3-step streaks; XY, resting and released are immediate. Released uses qpos <0.2 rad; fully open is near zero, below the measured ~0.645 rad closed-on-piece and ~0.78 rad closed-on-air apertures. No grasp is required by F2.

## One table of scenarios

| Scenario | Expected from preset | Intended | Measured result | Verdict | Evidence |
|---|---|---|---|---|---|
| Random absolute (100 seeds) | No success; forbidden contacts or >250 N board force terminate | No accidental success | Reasons {'bad_contact': 100}; success=0; instability=0; finger-contact steps=3 | OK | section7-evidence.json: random_section2_summary.json + random_episodes |
| Random delta (100 seeds) | No success; forbidden contacts or >250 N board force terminate | No accidental success | Reasons {'bad_contact': 40, 'time_limit': 60}; success=0; instability=0; finger-contact steps=0 | OK | section7-evidence.json: random_section2_summary.json + random_episodes |
| Random cartesian (100 seeds) | No success; forbidden contacts or >250 N board force terminate | No accidental success | Reasons {'time_limit': 100}; success=0; instability=0; finger-contact steps=0 | OK | section7-evidence.json: random_section2_summary.json + random_episodes |
| v3 reference e4→e5 | Clean place must succeed | Must succeed if clean | F2=True, clean_place=False, steps=146, opening height=6.205 mm | OPEN | evidence.json: s3_clean_e4_e5.json |
| v3 reference b7→g2 | Clean place must succeed | Must succeed if clean | F2=True, clean_place=False, steps=193, opening height=7.312 mm | OPEN | evidence.json: s3_clean_b7_g2.json |
| v3 reference c1→f6 | Clean place must succeed | Must succeed if clean | F2=True, clean_place=True, steps=207, opening height=-0.200 mm | OK | evidence.json: fu_reference_c1_f6.json |
| v3 reference f1→c6 | Clean place must succeed | Must succeed if clean | F2=True, clean_place=True, steps=230, opening height=-0.251 mm | OK | evidence.json: fu_reference_f1_c6.json |
| drop_10mm b7→g2 | Data only; intention unresolved | OPEN | F2=True; steps=178; tilt=0.016344°; XY distance=1.055 mm; D20=-69.476738; D21=462.995277 | OPEN | evidence.json: s3_drop_10mm_b7_g2.json |
| drop_10mm e4→e5 | Data only; intention unresolved | OPEN | F2=True; steps=138; tilt=0.001600°; XY distance=5.681 mm; D20=-13.860456; D21=245.896877 | OPEN | evidence.json: s3_drop_10mm_e4_e5.json |
| drop_20mm b7→g2 | Data only; intention unresolved | OPEN | F2=True; steps=178; tilt=0.001707°; XY distance=2.190 mm; D20=-69.631600; D21=470.593052 | OPEN | evidence.json: s3_drop_20mm_b7_g2.json |
| drop_20mm e4→e5 | Data only; intention unresolved | OPEN | F2=True; steps=138; tilt=0.001778°; XY distance=0.666 mm; D20=-14.071450; D21=261.275616 | OPEN | evidence.json: s3_drop_20mm_e4_e5.json |
| drop_30mm b7→g2 | Data only; intention unresolved | OPEN | F2=True; steps=176; tilt=0.001920°; XY distance=1.734 mm; D20=-69.725003; D21=474.391247 | OPEN | evidence.json: s3_drop_30mm_b7_g2.json |
| drop_30mm e4→e5 | Data only; intention unresolved | OPEN | F2=True; steps=137; tilt=0.001212°; XY distance=0.669 mm; D20=-14.279918; D21=277.655396 | OPEN | evidence.json: s3_drop_30mm_e4_e5.json |
| drop_5mm b7→g2 | Data only; intention unresolved | OPEN | F2=True; steps=184; tilt=0.001626°; XY distance=4.977 mm; D20=-69.550040; D21=470.873765 | OPEN | evidence.json: s3_drop_5mm_b7_g2.json |
| drop_5mm e4→e5 | Data only; intention unresolved | OPEN | F2=True; steps=139; tilt=0.001718°; XY distance=8.700 mm; D20=-13.837223; D21=236.725416 | OPEN | evidence.json: s3_drop_5mm_e4_e5.json |
| drop_80mm b7→g2 | Data only; intention unresolved | OPEN | F2=True; steps=165; tilt=0.001804°; XY distance=6.256 mm; D20=-69.894373; D21=436.047485 | OPEN | evidence.json: s3_drop_80mm_b7_g2.json |
| drop_80mm e4→e5 | Data only; intention unresolved | OPEN | F2=True; steps=122; tilt=0.000694°; XY distance=8.002 mm; D20=-14.402144; D21=223.078018 | OPEN | evidence.json: s3_drop_80mm_e4_e5.json |
| fu_drop_10mm c1→f6 | Data only; intention unresolved | OPEN | F2=True; steps=188; tilt=0.002232°; XY distance=0.889 mm; D20=-65.271804; D21=471.199009 | OPEN | evidence.json: s3_fu_drop_10mm_c1_f6.json |
| fu_drop_30mm c1→f6 | Data only; intention unresolved | OPEN | F2=True; steps=187; tilt=0.001403°; XY distance=0.688 mm; D20=-65.729659; D21=503.289766 | OPEN | evidence.json: s3_fu_drop_30mm_c1_f6.json |
| fu_drop_80mm c1→f6 | Data only; intention unresolved | OPEN | F2=True; steps=173; tilt=0.002185°; XY distance=7.924 mm; D20=-65.962015; D21=446.687403 | OPEN | evidence.json: s3_fu_drop_80mm_c1_f6.json |
| fu_push_then_open b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=True; steps=515; tilt=0.001541°; XY distance=28.576 mm; D20=-183.017225; D21=364.971121 | OPEN | evidence.json: s3_fu_push_then_open_b7_g2.json |
| fu_push_then_open e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=True; steps=150; tilt=0.001767°; XY distance=15.138 mm; D20=-16.116454; D21=82.712246 | OPEN | evidence.json: s3_fu_push_then_open_e4_e5.json |
| push_closed b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=0.002137°; XY distance=501.467 mm; D20=-186.451310; D21=99.166442 | OPEN | evidence.json: s3_push_closed_b7_g2.json |
| push_closed_base b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=0.002051°; XY distance=539.566 mm; D20=-173.558298; D21=183.313014 | OPEN | evidence.json: s3_push_closed_base_b7_g2.json |
| push_closed_base e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=0.002339°; XY distance=234.249 mm; D20=-47.272825; D21=184.716612 | OPEN | evidence.json: s3_push_closed_base_e4_e5.json |
| push_closed e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=True; steps=150; tilt=0.001767°; XY distance=15.138 mm; D20=-16.116454; D21=82.712246 | OPEN | evidence.json: s3_push_closed_e4_e5.json |
| push_closed_final b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=0.001891°; XY distance=28.576 mm; D20=-213.818750; D21=599.914323 | OPEN | evidence.json: s3_push_closed_final_b7_g2.json |
| push_closed_final e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=0.001046°; XY distance=28.854 mm; D20=-71.093669; D21=498.661684 | OPEN | evidence.json: s3_push_closed_final_e4_e5.json |
| push_open b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=103.984456°; XY distance=487.842 mm; D20=-177.340863; D21=96.470291; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_b7_g2.json |
| push_open_base b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=0.001446°; XY distance=551.544 mm; D20=-176.288485; D21=141.096356; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_base_b7_g2.json |
| push_open_base e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=0.001603°; XY distance=78.000 mm; D20=-33.653268; D21=148.521264; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_base_e4_e5.json |
| push_open e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=300; tilt=103.964727°; XY distance=52.780 mm; D20=-30.490812; D21=141.498184; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_e4_e5.json |
| push_open_final b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=0.001701°; XY distance=550.723 mm; D20=-575.609074; D21=512.546528; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_final_b7_g2.json |
| push_open_final e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=0.001681°; XY distance=77.276 mm; D20=-100.194356; D21=543.070668; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_final_e4_e5.json |
| push_open_fixed_floor b7→g2 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=103.964445°; XY distance=495.791 mm; D20=-625.047437; D21=248.203985; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_fixed_floor_b7_g2.json |
| push_open_fixed_floor e4→e5 | Closed gripper cannot satisfy released; after opening, intention unresolved | OPEN | F2=False; steps=1000; tilt=90.112450°; XY distance=108.174 mm; D20=-147.721669; D21=473.875392; upright on-target fully open push not realised | not realised | evidence.json: s3_push_open_fixed_floor_e4_e5.json |
| topple b7→g2 | Toppled piece near target must not succeed | Must not succeed | F2=True; steps=204; tilt=0.002469°; XY distance=26.729 mm; D20=-70.526355; D21=514.582385; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_b7_g2.json |
| topple e4→e5 | Toppled piece near target must not succeed | Must not succeed | F2=True; steps=161; tilt=0.017602°; XY distance=27.651 mm; D20=-14.866637; D21=278.362879; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_e4_e5.json |
| topple_e4_near_target e4→e5 | Toppled piece near target must not succeed | Must not succeed | F2=True; steps=152; tilt=0.002076°; XY distance=24.024 mm; D20=-16.746214; D21=280.362633; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_e4_near_target_e4_e5.json |
| topple_final b7→g2 | Toppled piece near target must not succeed | Must not succeed | F2=False; steps=300; tilt=103.965152°; XY distance=8.615 mm; D20=-77.252459; D21=521.712116 | OK | evidence.json: s3_topple_final_b7_g2.json |
| topple_final e4→e5 | Toppled piece near target must not succeed | Must not succeed | F2=False; steps=300; tilt=0.001598°; XY distance=40.985 mm; D20=-25.580293; D21=351.484137; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_final_e4_e5.json |
| topple_head_bump b7→g2 | Toppled piece near target must not succeed | Must not succeed | F2=False; steps=300; tilt=103.965229°; XY distance=39.201 mm; D20=-77.754159; D21=527.590842; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_head_bump_b7_g2.json |
| topple_head_bump e4→e5 | Toppled piece near target must not succeed | Must not succeed | F2=False; steps=300; tilt=0.002136°; XY distance=39.711 mm; D20=-25.106870; D21=356.007289; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_head_bump_e4_e5.json |
| topple_left_finger e4→e5 | Toppled piece near target must not succeed | Must not succeed | F2=False; steps=300; tilt=103.964573°; XY distance=72.184 mm; D20=-29.035095; D21=344.632991; requested near-target topple not realised (recovered or outside 30 mm) | not realised | evidence.json: s3_topple_left_finger_e4_e5.json |
| D20 six earlier drop comparisons | Known D20 can pay short trajectories more; comparator must be clean | Push/drop remain OPEN | Six advantages vs non-clean references; duration decomposition retained in REPORT.md; comparator invalid for clean-place claim | RM | REPORT.md: Six earlier drops; RM, old reward, mechanism unconfirmed; known D20, not a new finding |
| Clean-reference reward comparison fu_drop_10mm | Record both returns; do not decide drop | OPEN | ΔD20=+0.146083; ΔD21=-25.046726; length=188 vs 207 | OPEN | section7-evidence.json: all13_release_events.json |
| Clean-reference reward comparison fu_drop_30mm | Record both returns; do not decide drop | OPEN | ΔD20=-0.311772; ΔD21=+7.044031; length=187 vs 207 | OPEN | section7-evidence.json: all13_release_events.json |
| Clean-reference reward comparison fu_drop_80mm | Record both returns; do not decide drop | OPEN | ΔD20=-0.544128; ΔD21=-49.558332; length=173 vs 207 | OPEN | section7-evidence.json: all13_release_events.json |
| Open-pad board-force ramp | Only >250 N combined robot-board force terminates | Below/equal: must continue; above: must terminate | Step 1125: 234.855268 N, allowed; step 1126: 251.517261 N, bad_contact; no ≤250 N bad_contact | OK | board-force-steps.tsv: rows 1125–1126 |
| Five gripper cycles / D25 | Finger-finger contact must not terminate | Must continue | Five cycles, 15 finger-contact steps, no termination; held ends 121; board contact 122 | OK | evidence.json: s4_gripper_cycles.json |
| D21 first-grasp bonus during cycling | +5 only once per episode on first held step | One initial bonus; no repeat without held | Episode bonus fires=1 at [80]; cycle fires=0; held False→True in cycles=0 | OK | section7-evidence.json: cycle_bonus_contact_check.json.summary |
| T2 3mm | >5 mm or >0.26 rad must latch; below must not | Must not latch | Sampled displacement=3.000000 mm; tilt=0.000027 rad; latch=False; restored latch=False, undisturbed=True | OK | section7-evidence.json: section5.json |
| T2 8mm | >5 mm or >0.26 rad must latch; below must not | Must latch | Sampled displacement=7.999999 mm; tilt=0.000033 rad; latch=True; restored latch=True, undisturbed=True | OK | section7-evidence.json: section5.json |
| T2 tilt .20 | >5 mm or >0.26 rad must latch; below must not | Must not latch | Sampled displacement=0.244393 mm; tilt=0.090112 rad; latch=False; restored latch=False, undisturbed=True | OK | section7-evidence.json: section5.json |
| T2 tilt .30 | >5 mm or >0.26 rad must latch; below must not | Must latch | Sampled displacement=0.312347 mm; tilt=0.217290 rad; latch=False; restored latch=False, undisturbed=True; initial 0.30 rad relaxed below threshold before sampling; no subsequent placement-success test | OPEN | section7-evidence.json: section5.json |
| T2 tilt .40 | >5 mm or >0.26 rad must latch; below must not | Must latch | Sampled displacement=0.312268 mm; tilt=0.341500 rad; latch=True; restored latch=True, undisturbed=True | OK | section7-evidence.json: section5.json |
| Cartesian home hold | Hold should remain stable and legal | Must not terminate | steps=300; ΔXYZ mm=(-0.025, -2.235, -10.498); peak tool tilt=0.576400°; rejects=0; exact stalls=0; ≤1 µm/no-reject=262; clips box/outer/inner=0/0/0; 195.6 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian hover hold | Hold should remain stable and legal | Must not terminate | steps=300; ΔXYZ mm=(-3.480, -1.268, +5.922); peak tool tilt=0.033708°; rejects=0; exact stalls=0; ≤1 µm/no-reject=264; clips box/outer/inner=0/0/0; 220.7 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian x-1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=150; ΔXYZ mm=(-388.786, -49.129, +0.781); peak tool tilt=0.038278°; rejects=0; exact stalls=0; ≤1 µm/no-reject=50; clips box/outer/inner=112/0/0; 174.9 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian x+1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=150; ΔXYZ mm=(+313.468, -44.596, -0.057); peak tool tilt=0.042221°; rejects=0; exact stalls=0; ≤1 µm/no-reject=54; clips box/outer/inner=123/0/0; 210.5 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian y-1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=37; ΔXYZ mm=(-23.572, -281.151, -20.411); peak tool tilt=0.183213°; rejects=0; exact stalls=0; ≤1 µm/no-reject=0; clips box/outer/inner=0/0/4; 161.3 steps/s; bad_contact, shoulder_link↔wrist_1_link | FT | section7-evidence.json: section6.json; yminus_geometry.json |
| Cartesian y+1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=150; ΔXYZ mm=(-25.501, +377.764, -21.577); peak tool tilt=0.098606°; rejects=0; exact stalls=0; ≤1 µm/no-reject=35; clips box/outer/inner=97/99/0; 191.4 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian z-1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=150; ΔXYZ mm=(-6.367, -9.073, -90.328); peak tool tilt=0.159039°; rejects=0; exact stalls=0; ≤1 µm/no-reject=72; clips box/outer/inner=138/0/0; 251.3 steps/s | OK | section7-evidence.json: section6.json |
| Cartesian z+1 | Guarded box-face push should remain legal; clipping/holding allowed | Must not terminate | steps=150; ΔXYZ mm=(-15.336, -33.951, +298.070); peak tool tilt=0.103895°; rejects=0; exact stalls=0; ≤1 µm/no-reject=52; clips box/outer/inner=113/0/0; 248.9 steps/s | OK | section7-evidence.json: section6.json |
| T2 disturbed goal, sparse | T2 success must be blocked after obstacle disturbance | Must not report success | reason=success_obstacles_disturbed; info.is_success=False; actual step reward=0.000000000; compute_reward=1.000000000 | OK | section7-evidence.json: t2_reward_check.json |
| T2 disturbed goal, shaped | T2 success must be blocked after obstacle disturbance | Must not report success | reason=success_obstacles_disturbed; info.is_success=False; actual step reward=-0.196597716; compute_reward=-0.000007451 | OK | section7-evidence.json: t2_reward_check.json |
| T2 disturbed goal, staged | T2 success must be blocked after obstacle disturbance | Must not report success | reason=success_obstacles_disturbed; info.is_success=False; actual step reward=-0.196590266; compute_reward=-0.000007451 | OK | section7-evidence.json: t2_reward_check.json |
| T2 sparse HER/compute_reward (F3) | Extra success conditions ignored by known HER path | Known F3; HER off | Returns 1 at same goals despite disturbed-obstacle info; actual sparse step returns 0 | RM | section7-evidence.json: t2_reward_check.json; known F3, not a new finding |

## Expectation mismatches and limits

- The e4→e5 and b7→g2 original v3 references succeeded under F2 but were not clean v3 places (opening heights 6.205 and 7.312 mm). They are invalid clean comparators. c1→f6 and f1→c6 supply verified clean references.

- Push-then-open succeeded without any held, pinched or grasp_success state in both pairs. This is measured data; the push verdict remains OPEN. Fully open upright on-target pushes and the e4→e5 near-target topple were not realised and attempts were stopped. Recovery after a transient tilt is not false success.

- The 0.30 rad obstacle fixture crossed the tilt threshold before integration but relaxed to 0.217290 rad at referee sampling. No latch occurred. This conflicts with a continuous-history reading of the preset, while matching the implementation's macro-step sampling. The >0.26 rad sampled case (0.341500 rad) did latch and retained it after restoration. The 0.30 rad row is OPEN: no later placement-success test was performed, so an FS verdict is not established. These were controlled pose perturbations, not physical gripper brushes.

- Zero XYZ command showed settling drift: 10.733777 mm from reset home and 6.985189 mm after hover arrival. The holds were legal and did not terminate; OK refers to referee response, not a claim of zero drift. No numerical drift acceptance threshold was preset.

- The −Y push terminated after 37 of the requested 150 steps, before the box face. This violates the guarded-control expectation (FT). The referee detected a genuine shoulder_link↔wrist_1_link contact; this is not a spurious contact detection and differs from the documented upper_arm↔wrist_2 pair. The inner radial guard was active and all captured IK targets matched the clipping calculation.

- Reward advantages do not classify push/drop as improper. The six earlier D20 comparisons use non-clean references and retain the label **RM, old reward, mechanism unconfirmed; known D20**. Clean c1→f6 comparisons show the 10 mm drop exceeds clean D20 by 0.146083 and the 30 mm drop exceeds clean D21 by 7.044031. Intent remains OPEN; no drop limit is recommended.

## −Y geometry at contact

Push-local step 37 is episode step 94, after 57 hover-setup steps. All dimensions below are metres. Radius is measured about the model arm base axis, confirmed at (0,0).

| Quantity | Value |
|---|---:|
| Pinch x after step 37 | 0.014205247478 |
| Pinch y after step 37 | 0.200842893075 |
| Pinch z after step 37 | 0.863853548065 |
| Radius after step 37 | 0.201344621867 |
| Inner limit at actual z | 0.196687871161 |
| Actual radius minus inner limit | 0.004656750706 |

Inner clips: push steps **34, 35, 36, 37**; episode steps **91, 92, 93, 94**. The step-37 IK target is (0.014700798293, 0.196166201987, 0.863806210779); its inner limit is 0.196716273533 m. Target clipping uses the pre-step target z, not the post-step measured z. Contact: shoulder_link↔wrist_1_link, 172.758 N and 0.4983 mm penetration. All 37 captured targets match. The measured-radius margin alone does not establish a safe arm pose.

## Release-event correction, all 13 drops

Opening height is the base height immediately before the first opening command. Last pad contact and first qpos <0.2 sample are measured at 2 ms resolution, with substeps numbered 1–20 inside each 40 ms environment step. Values below supersede the coarse last-pad and crossing measurements in the earlier report. Heights are relative to the board surface; negative values are penetration. Replays reproduce the recorded returns.

| Pair | Nominal mm | Opening height mm | Last pad step/substep | Last pad height mm | qpos crossing step/substep | Crossing height mm |
|---|---:|---:|---|---:|---|---:|
| b7→g2 | 5 | 5.916858 | 179/4 | 5.837286 | 182/5 | -0.074660 |
| b7→g2 | 10 | 11.597598 | 174/4 | 11.651361 | 177/5 | 0.000740 |
| b7→g2 | 20 | 22.414709 | 173/4 | 22.438610 | 176/5 | -0.088489 |
| b7→g2 | 30 | 32.489384 | 171/4 | 32.510067 | 174/5 | -0.323987 |
| b7→g2 | 80 | 81.045533 | 159/4 | 80.968285 | 162/5 | 2.510393 |
| c1→f6 | 10 | 10.611976 | 184/4 | 10.284031 | 187/5 | -0.008797 |
| c1→f6 | 30 | 30.163372 | 182/4 | 29.796923 | 185/5 | -0.062382 |
| c1→f6 | 80 | 79.380060 | 166/4 | 79.102362 | 169/5 | -2.089595 |
| e4→e5 | 5 | 3.876889 | 134/4 | 3.724719 | 137/4 | -0.107920 |
| e4→e5 | 10 | 10.637248 | 133/4 | 10.403896 | 136/4 | -0.027394 |
| e4→e5 | 20 | 20.210708 | 133/4 | 19.953573 | 136/4 | -0.046885 |
| e4→e5 | 30 | 29.991711 | 132/4 | 29.723728 | 135/4 | -0.180280 |
| e4→e5 | 80 | 80.103839 | 116/4 | 80.060804 | 119/4 | 4.268492 |

## Cycling timing and first-grasp accounting

One D21 +5 first-grasp bonus at episode step 80; zero during the five cycles. Held is never True during cycle steps 121–254, and has zero False→True transitions there.

| Step | Height mm | Left/right pad contact | Board contact | Held |
|---|---:|---|---|---|
| 119 | 30.336822 | True/True | False | True |
| 120 | 29.372657 | True/True | False | True |
| 121 | 21.565164 | False/False | False | False |
| 122 | -1.336729 | False/False | True | False |
| 123 | -0.267958 | False/False | True | False |

First board contact: step 122, substep 19 (1-based), 78.000000 ms after step 120. Zero-initial-velocity free fall from the setup height: 77.384167 ms. The measured fall is consistent; two macro-steps represent 80 ms, not 40 ms.

## Budget and contact-depth follow-up

b7→g2 push used constructor budget 1000; opening 512, success 515. Board ramp used constructor budget 4000; termination 1126. Cycling set the existing instance budget to 1000 after the lift prefix. No success, contact or streak counters were reset or overridden. Section 6 used budget 1000 so setup did not consume the requested hold/push duration.

Across 118 first presence triggers, penetration min/median/max = 0.003195529/5.447746506/40.836291553 mm. Minimum pair: <ur_shoulder_link>#geom8 ↔ ur_gripper_left_pad1#geom56. All recorded first-trigger pairs are neither finger–finger nor pad–piece.

## T2 success accounting and known F3

`success_obstacles_disturbed` is a termination-reason value, not an info key. The measured fixture has base placement `_is_success()` true (the underlying env terminates on it) but T2 `info["is_success"]` false. The configured Monitor columns are terminated_reason, gripper_self_contact and ik_rejects; they do not include is_success. Monitor retains the relabelled termination reason and does not convert it into success. The optional pickup_success column is a separate D21 criterion, and pickup_obs rejects T2. The code assigning the reason is:

```python
if base_success and ever_disturbed and info.get("terminated_reason") == "success":
    info["terminated_reason"] = "success_obstacles_disturbed"
```

Source: robochess/tasks/t2_obstacle_pick_and_place.py:486–487. Actual sparse reward is 0; sparse compute_reward is 1 (known F3, HER off). Shaped/staged paths do not read the latch and pay distance shaping, not a success bonus; their measured returns are in the summary.

## Git provenance

Before this report commit the tree was clean at f4581ae. The CI files are already committed there: `.github/workflows/ci.yml` pins Ubuntu 24.04, updates checkout/setup-python majors, and makes CPU torch indexes explicit; `requirements-ci.txt` adds pinned SB3 and sb3-contrib. Recorded author/committer: Zeeshaan Ebrahim; co-author: Claude Sonnet 5. This reporting pass did not modify, commit or discard those CI edits. a8517ba contains only the three report/evidence files. Branch base 8f122a3 predates those report commits and includes teammate probe evaluation scripts/results; it is distinct from the simulation baseline a37c8ab.

## Reproduction and stop point

section7-evidence.json includes complete random episode summaries and contact ledger, release samples, cycling substeps, latch/control measurements, the 37-step −Y geometry trace, and scratch replay scripts as text. The initial sections 5/6 scratch runner had a tool-tilt instrumentation error; the corrected rerun completed with pipeline exit 0. The final −Y replay also completed with exit 0 and asserted 37 steps, four inner clips, and matching targets. No unit suite was run for these report-only edits. Stopped after section 7; no source fixes and no push.
