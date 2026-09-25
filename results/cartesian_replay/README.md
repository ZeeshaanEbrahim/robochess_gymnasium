# Scripted expert replay through Cartesian control (25 Sep 2026)

Diagnostic gate only. No environment, referee, action mapping, schedule, or threshold was changed. Run with the project's Python 3.11 venv:

```sh
python scripts/replay_expert_cartesian.py --out results/cartesian_replay/replay_200.json
```

This uses the established delta-replay sample: ten starts, target `e5`, seed 0, an absolute-mode reference budget of 350 steps, and the training builder's T1 Cartesian mode at 200 steps. For every reference macro-step, the harness converts the expert's six commanded arm joint angles to a commanded pinch position by forward kinematics. It clips the displacement from the replay env's *measured* pinch position toward that point to the Cartesian mode's existing per-axis 20 mm step. The recorded gripper command and the reference phase timing are retained. The replay goes through `scripts/train.py`'s `build_env()` and `env.step()`; it does not alter the env state directly. Thus this tests a timed recording under the Cartesian control scheme, not all possible Cartesian controllers.

## Measured result

| Start | Reference | Replay | Grasp ever | End step / 200 | Replay end |
| --- | --- | --- | --- | --- | --- |
| a1 | success | fail | no | 56 | bad_contact, lift |
| a8 | success | fail | no | 78 | bad_contact, close |
| h1 | success | fail | no | 56 | bad_contact, lift |
| h8 | success | fail | no | 61 | piece_off_board, descend |
| d1 | bad_contact at step 37 | fail | no | 37 | command recording ended, descend |
| d8 | success | fail | no | 78 | bad_contact, close |
| b4 | success | fail | yes | 75 | bad_contact, lift |
| g6 | success | fail | yes | 168 | bad_contact, release |
| c7 | success | fail | yes | 156 | bad_contact, place |
| f2 | success | fail | no | 57 | bad_contact, lift |

**0/10 replay successes; 3/10 registered grasps.** Nine reference episodes succeed. `d1` is not a valid reference success under the current referee: it ends at step 37, so its Cartesian replay has only 37 commands and cannot reach the task end. All ten Cartesian replays have zero IK rejects. No replay consumes the 200-step budget; the failure is earlier than budget exhaustion.

## Diagnosis

The reference is a waypoint-jump controller with phases timed by its faster absolute-mode run. The Cartesian interface limits each requested world-axis displacement to 20 mm per 40 ms macro-step and further limits the joint change. At the end of `approach`, the measured pinch-to-command gap is 0.133–0.346 m on the nine valid references. When `descend` ends it remains 0.004–0.241 m. On `a1`, `h1`, and `f2`, the replay closes with the pinch 0.067–0.235 m from the recorded command. The schedule has advanced while the Cartesian arm is still approaching. `a8` and `d8` arrive much closer (0.003–0.005 m at close) but still terminate in the close phase without a registered grasp.

At each of the eight `bad_contact` terminations, the final MuJoCo contact list contains `ur_gripper_right_pad1` touching `ur_gripper_left_pad1`. There is no robot-to-board force reading at those ends. The referee classifies a contact between different robot bodies as self contact, so the pad pair is sufficient to account for `bad_contact`. It happens even after a registered grasp on `b4`, `g6`, and `c7`; their later failure cannot be described solely as an early empty-gripper close. `h8` is a separate `piece_off_board` failure during descend. The output JSON contains per-step phase and pinch-to-command gaps plus terminal contact pairs for inspection.

This replay gate **fails**. The data do not establish that Cartesian control cannot solve T1. A closed-loop controller may allocate different time to each phase and choose a different path. No schedule, referee rule, threshold, or calibration is recommended from this result alone. Under the repo's diagnosis/fix separation rule, this pass stops here.
