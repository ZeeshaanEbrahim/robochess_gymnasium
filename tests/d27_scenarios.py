"""Fresh, self-contained Cartesian-mode scenarios for D27 tests.

Built only on the env's own Cartesian action interface (pinch-site displacement
+ gripper) and the scripted expert's two published heights; no code from the
adversarial sweep. Every scenario returns a `Trace` of per-step infos so a test
can assert on the recorded release height as well as on success.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402

BRAKE_FRACTION = 0.4
BRAKE_MIN_M = 1.5e-3
LATERAL_TOL_M = 2.0e-3
VERTICAL_TOL_M = 1.0e-3


@dataclass
class Trace:
    infos: list = field(default_factory=list)
    terminated: bool = False

    @property
    def last(self) -> dict:
        return self.infos[-1]


class Driver:
    """Cartesian-mode controller: braked moves toward a pinch target."""

    def __init__(self, start: str, target: str, budget: int = 900, seed: int = 0,
                 env_factory=None):
        """`env_factory(base, start, target)` may wrap the T1 task further (e.g. in
        PickupReward); it must return an env whose `.unwrapped` is the RoboChessEnv."""
        self.base = RoboChessEnv(action_mode="cartesian", max_episode_steps=budget)
        task = T1PickAndPlaceTask(self.base, start_square=start, target_square=target)
        self.env = env_factory(task) if env_factory else task
        self.on_step = None  # optional callback(out) after every env step
        self.env.reset(seed=seed, options={"start_square": start, "target_square": target})
        b = self.base
        self.piece_xy = b._achieved_goal()[:2].astype(float).copy()
        self.goal_xy = b._desired_goal[:2].astype(float).copy()
        self.trace = Trace()
        span = b._ctrl_high[6] - b._ctrl_low[6]
        self._grip_action = {
            "open": -1.0, "closed": 1.0,
        }
        self._span = span

    # ------------------------------------------------------------ primitives
    @property
    def pinch(self) -> np.ndarray:
        return self.base.data.site_xpos[self.base._pinch_sid].copy()

    @property
    def piece_height_m(self) -> float:
        return float(self.base._achieved_goal()[2] - self.base._board_surface_z)

    def step(self, disp_m, grip: str):
        """One env step: pinch displacement in metres (clipped to the step), gripper."""
        if self.trace.terminated:
            return self.trace.last
        disp = np.clip(np.asarray(disp_m, dtype=np.float64) / self.base.cartesian_step, -1.0, 1.0)
        action = np.concatenate([disp, [self._grip_action[grip]]]).astype(np.float32)
        out = self.env.step(action)
        _, _, term, trunc, info = out
        if self.on_step is not None:
            self.on_step(out)
        self.trace.infos.append(info)
        self.trace.terminated = bool(term or trunc)
        return info

    def goto(self, target, grip: str, max_steps: int = 120, fast: bool = False) -> bool:
        """Braked move until inside tolerance (2 mm lateral, 1 mm vertical)."""
        target = np.asarray(target, dtype=np.float64)
        for _ in range(max_steps):
            to = target - self.pinch
            if np.hypot(to[0], to[1]) <= LATERAL_TOL_M and abs(to[2]) <= VERTICAL_TOL_M:
                return True
            dist = float(np.linalg.norm(to))
            if fast:
                disp = to
            else:
                mag = max(BRAKE_MIN_M, min(20.0e-3, BRAKE_FRACTION * dist))
                disp = to / dist * mag
            self.step(disp, grip)
            if self.trace.terminated:
                return False
        return False

    def hold(self, grip: str, steps: int):
        for _ in range(steps):
            self.step(np.zeros(3), grip)
            if self.trace.terminated:
                break

    # ------------------------------------------------------------ composites
    def grasp_and_lift(self, xy=None) -> None:
        """Approach, descend, close until the aperture is steady, lift to transit height."""
        px, py = self.piece_xy if xy is None else xy
        self.goto([px, py, se.CLEAR_Z], "open")
        self.goto([px, py, se.GRASP_Z], "open")
        prev = float(self.base.data.qpos[self.base._grip_qadr])
        steady = 0
        for _ in range(60):
            self.step(np.zeros(3), "closed")
            now = float(self.base.data.qpos[self.base._grip_qadr])
            steady = steady + 1 if abs(now - prev) < 0.002 else 0
            prev = now
            if steady >= 2:
                break
        self.goto([px, py, se.CLEAR_Z], "closed")

    def carry_to(self, xy, height_m: float, fast: bool = False) -> None:
        """Traverse at transit height, then lower until the piece BASE is `height_m`
        above the board (the pinch-to-piece offset is measured live)."""
        x, y = xy
        self.goto([x, y, se.CLEAR_Z], "closed")
        in_hand = self.piece_height_m  # piece base above board while carried at CLEAR_Z
        pinch_z = float(self.pinch[2]) - in_hand + height_m
        self.goto([x, y, pinch_z], "closed", fast=fast)

    def open_and_settle(self, steps: int = 60) -> None:
        """Open the gripper with zero commanded motion until the episode ends or `steps`."""
        self.hold("open", steps)

    def place_and_release(self, xy=None, height_m: float = 0.0015, fast: bool = False) -> None:
        self.carry_to(self.goal_xy if xy is None else xy, height_m, fast=fast)
        self.open_and_settle()


def push_then_open(dr: Driver, back_m: float = 0.06, stop_err_m: float = 0.004) -> None:
    """Never grasp: slide the piece onto its target with a closed, empty gripper,
    then open. Approach from the -y side of the piece at grasp height (target is
    on the +y side for e4->e5), push along +y until the piece is within
    `stop_err_m` of the goal in XY, stop, open, and let it settle."""
    px, py = dr.piece_xy
    gx, gy = dr.goal_xy
    u = np.array([gx - px, gy - py]) / np.hypot(gx - px, gy - py)
    behind = np.array([px, py]) - u * back_m
    dr.goto([behind[0], behind[1], se.CLEAR_Z], "closed")
    dr.goto([behind[0], behind[1], se.GRASP_Z], "closed")
    for _ in range(300):
        piece = dr.base._achieved_goal()[:2].astype(float)
        err = np.array([gx, gy]) - piece
        if np.hypot(*err) <= stop_err_m or dr.trace.terminated:
            break
        dr.step(np.array([u[0], u[1], 0.0]) * 3.0e-3, "closed")  # 3 mm/step push
    dr.hold("closed", 5)
    dr.hold("open", 60)


def fast_set_down(dr: Driver, hover_above_m: float = 0.03) -> None:
    """Carry to a point `hover_above_m` above the goal (piece base), then descend at
    the MAXIMUM Cartesian step (full-scale dz every step) onto the board while
    still pinched, then open. The descent is not braked, so the piece meets the
    board at the highest speed the interface allows."""
    dr.carry_to(dr.goal_xy, hover_above_m)
    for _ in range(20):
        if dr.trace.terminated or dr.piece_height_m <= 0.0005:
            break
        dr.step([0.0, 0.0, -dr.base.cartesian_step], "closed")  # full-scale dz
    dr.open_and_settle()


def regrasp(dr: Driver, off_target_offset_m=(2 * 0.078, 0.0)) -> None:
    """Release on the board OFF target (low, clean), then re-grasp the piece where it
    now lies and carry it to the target for a low release."""
    off = dr.goal_xy + np.asarray(off_target_offset_m)
    dr.carry_to(off, 0.0015)
    dr.open_and_settle(30)
    dr.mid = dict(dr.trace.last)  # state after the first, off-target release
    dr.goto([off[0], off[1], se.CLEAR_Z], "open")
    piece_xy = dr.base._achieved_goal()[:2].astype(float)
    dr.grasp_and_lift(xy=piece_xy)
    dr.place_and_release(height_m=0.0015)
