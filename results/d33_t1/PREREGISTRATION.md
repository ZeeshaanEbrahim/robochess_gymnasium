# D33: the T1 cell again, with success that requires a real lift (pre-registered before any D33 run)

Question: if a placement only counts after the piece has been lifted well off the board while held, does PPO with the D29 configuration and the D31 recipe still learn the fixed move e4:e5?

Why: RS32 found that 80% to 94% of placements never lifted the piece more than 10 mm (results/rs32_48_pairs/eval/diagnosis/DIAGNOSIS.md, X2): the policies place by dragging. A dragging policy can pass the D32 T2 cell without ever clearing an obstacle. D33 asks whether a lifting policy can be learned at all under the same recipe.

## Cell
Identical to the D32 T1 control (configs/d32_t1_e4e5_seed*.yaml, commit 53ccc06): task t1, fixed move e4:e5, no obstacles, cartesian, max_episode_steps 300, D29 configuration, D31 recipe (n_envs 4, linear learning-rate decay, ppo_ent_coef 0.01), 3,000,000 steps. The only changes are two config keys, `place_require_lift: true` and `place_lift_latch_m: 0.020`, and run_name. Seeds 0, 100, 200, 300, 400. The configs are configs/d33_t1_e4e5_seed{0,100,200,300,400}.yaml; a diff against the control config showing exactly that is in the commit that adds them.

## What the flag does
- Latch: set the first time the working piece is at least `place_lift_latch_m` (20 mm) above its resting height WHILE BOTH PADS ARE ON IT (pinched; held is a subset). It clears only at reset. Lift is `piece height above the board` (0 mm at rest).
- With the flag on, the latch must be set before any of these four, which change together because they all hang on the env's `release_ok`: S2 entry, the placed bonus (+10), the success lump (1020), and the success flag (info["is_success"] and terminated_reason "success"). Where each is computed (this branch, after the code commit):

| Quantity | File and line |
|---|---|
| lift latch set | robochess/envs/robochess_env.py:1380-1382 (inside the pinched branch of step) |
| lift latch cleared | robochess/envs/robochess_env.py:942 (reset) |
| `release_ok` gated on the latch | robochess/envs/robochess_env.py:1118-1130 (`_release_ok`) |
| success flag and termination reason | robochess/envs/robochess_env.py:1135 (`_success_terms`), :1200 (`_is_success`), :1261 (`info["is_success"]`), :1390-1398 (`success`, reason "success") |
| S2 entry | robochess/tasks/t1_pick_and_place.py:1334-1342 (`in_s2` reads info["release_ok"]) |
| placed bonus | robochess/tasks/t1_pick_and_place.py:1382 (`placed_fired`, follows S2) |
| success lump | robochess/tasks/t1_pick_and_place.py:1388 (reads info["is_success"]) |
| T2 wrapper (not used in D33) | robochess/tasks/t2_obstacle_pick_and_place.py:488 and :501 only AND obstacles into is_success and relabel the reason |

- The existing one-time lift bonus (+5) is paid once at the latch, not at 30 mm held for 5 steps (t1_pick_and_place.py:1367-1377).
- Unchanged: the 15 mm release-height requirement (D27), the shaped lift term (2.0 x lift_frac, capped at 30 mm), the near-target lowering term, the posture gate, the carry potential, all other weights.
- Flag off is byte-identical: the pre-edit and post-edit trace hashes (T1 D config, flag-off T2) and the D29 flag-off hashes are unchanged; the full suite passes.
- The latch threshold, 20 mm, was fixed before any run. It is above RS32's 90th percentile of maximum lift among placed episodes (9.2 to 12.1 mm per seed, results/rs32_48_pairs/eval/diagnosis/diag_addendum_tables_output.txt, X2), below the 30 mm cap of the shaped lift term, and below the largest lift RS32 placed episodes reached (36.2 mm). It is also close to RS32's 99th percentile of that maximum (20.3 to 26.0 mm), which is the reason the prediction below is what it is.

## Evaluation and rule (identical to D32)
- Final model, 200 stochastic episodes per seed, eval seed 20000, frozen VecNormalize statistics, scripts/eval_probe.py, fixed pair e4:e5.
- Success is terminated_reason "success". A seed learns at >= 100 of 200. The claim "PPO with the D29 configuration and the D31 recipe learns the lift-required cell" holds if at least 3 of 5 seeds learn. No threshold changes after any result.

## Reported, not gated
- The D32 control's success next to D33's, per seed and pooled (Wilson 95% intervals).
- Share of placed episodes whose maximum piece lift exceeded 10 mm and exceeded 50 mm (max_lift_mm from the eval output, placed = terminated_reason "success").
- Why failing episodes fail, from fields the standard eval already records: toppled = terminated_reason "piece_off_board", or the piece tilt in S1 above 45 degrees (place.max_s1_tilt, the upper end of the D29 posture gate); bad contact = terminated_reason "bad_contact"; stuck low = time limit with max_lift_mm below 20 mm and not toppled (the latch was never reached); lifted but did not finish = time limit with max_lift_mm at or above 20 mm and not toppled. These are counts of mutually exclusive classes in that order. max_lift_mm is the piece lift at any time, not only while pinched, so "lifted but did not finish" can include a piece that was knocked up without being held; that is a known approximation.

## Prediction (not a gate)
Success will fall below the control, because lifting 20 mm while pinched is rare under this recipe (RS32: it is about the 99th percentile of the maximum lift among placed episodes). This is a finding either way.

## Decision rule (fixed now)
If D33 meets the rule on at least 3 of 5 seeds, T2 switches to the D33 reward in a new pre-registered cell. Otherwise dragging is reported as a finding and T2 stays on the D32 cell.

## Limitations
- The control and D33 runs come from different SHAs (control: branch recipe, 53ccc06; D33: branch d33). D33 additionally contains the D32 T2 code change (75b63ea), which does not touch the T1 path (traces unchanged).
- Five seeds per arm; 3 of 5 is coarse. 3,000,000 steps may be too few to find a rare behaviour; a null result is not evidence that lifting cannot be learned.
- The latch is a one-way switch: after it is set, the policy is free to drag again for the rest of the episode. The flag requires that a lift happened, not that the piece was carried.
- Changing the +5 lift bonus from 30 mm held for 5 steps to the 20 mm latch is part of the same bundle; no attribution between the latch and the bonus change is claimed.
- The eval_probe `--d32` recorder (branch t2-d29) is not on this branch; the reported quantities above come from the standard eval output.
