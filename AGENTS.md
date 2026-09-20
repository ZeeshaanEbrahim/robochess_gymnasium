# Working agreements for coding agents

This file is the agent-facing contract for RoboChessGymnasium. It applies to
Codex, Claude Code, and anything else driving this repo.

## Environment

- Python env: venv at `~/venvs/robochess` (Python 3.11.15). Activate it before
  running anything; the system Python is 3.10.12 and is deliberately untouched.
- `requirements.txt` is the **frozen experiment set** and pins a CUDA torch
  build (~4.5 GB). `requirements-ci.txt` is the CPU-only set CI installs. Never
  install `requirements.txt` on a machine without a GPU.
- Rendering: `MUJOCO_GL` is unset locally (GLFW via WSLg). On a headless box,
  export `MUJOCO_GL=osmesa`.
- Vendor meshes under `assets/ur5e/assets/` and `assets/robotiq_2f85/assets/`
  are gitignored. Restore them with `scripts/fetch_assets.sh`, which copies from
  `~/reference_assets/mujoco_menagerie`. Nothing loads `assets/scene.xml` until
  this has run.

## Vendor assets

Copy mesh files **out** of the menagerie checkout into `assets/`; never edit
files in `~/reference_assets/mujoco_menagerie`, and never reference them in
place. The model XML files alongside the meshes ARE tracked and carry local
modifications (2F-85 tool mount on the UR5e wrist, matched `<option>` settings)
— do not overwrite them with vendor XML.

## Tests

Run the suite exactly as CI does:

    pytest tests/ -W error::UserWarning

CI promotes `UserWarning` to an error. A plain `pytest` run reported
"97 passed, 1 warning" for days while CI was red. Treat any nonzero warning
count in a summary as a finding, not noise. For environment-sensitive failures,
reproduce in a throwaway venv built from `requirements-ci.txt` only (Python
3.11, `MUJOCO_GL=osmesa`).

## Git

**Never run `git push`.** Standing decision (29 Aug 2026): agents commit with
clear messages; Zeeshaan pushes manually from his own authenticated environment
after reviewing each commit. Do not set up a PAT, SSH key, deploy key, or
`gh auth` for this purpose, and do not spend time on a failed push — just
commit and say it is ready to push.

## How to work

- **Diagnosis and fixes are separate passes.** When an anomaly surfaces,
  root-cause it fully and report, then stop. Do not fix in the same pass.
  Mixing the two previously produced four failed fix attempts on D7 before the
  real cause (no gravity compensation on the actuators) was found.
- **Re-verify by measurement.** Anything inferred from a calculation, a single
  example, or a code comment must be confirmed by actually running the behavior
  it predicts before it is trusted or committed. A self-consistent formula or
  comment is not evidence.
- **Report plainly.** "Zero effect" and "failed verification" are reportable
  results. Do not round toward a plausible-sounding but unconfirmed
  explanation.
- **Ask before anything not covered by the instructions** — in particular, do
  not pick threshold or calibration values unilaterally.
- Work in numbered sections where the task allows, stopping to report the
  verification result after each. Default breakdown: (1) skeleton/structure,
  (2) reward/logic layer, (3) unit tests + coverage, (4) scripted-expert
  regression check, (5) short smoke test.

## Source of truth

The decisions table in `ROADMAP.md` (section 8) is authoritative. `DECISIONS.md`
mirrors it; if they drift, reconcile `DECISIONS.md` to the roadmap, never the
reverse.
