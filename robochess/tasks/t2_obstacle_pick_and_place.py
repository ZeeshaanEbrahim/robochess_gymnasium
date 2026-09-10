"""T2: pick-and-place with bystander obstacle pieces on the board.

T2 is T1 plus obstacles, so it extends :class:`T1PickAndPlaceTask` rather than
re-deriving start/target selection, teleporting and goal placement. That is
still composition with respect to the environment -- ``RoboChessEnv`` is
wrapped, never subclassed, exactly as in T1 -- the subclassing here is one
wrapper extending another.

Everything T2 adds lives in this module. ``RoboChessEnv`` and
``t1_pick_and_place.py`` are untouched by T2, including the observation
extension below, which is applied at the wrapper level rather than by
widening the base env's ``_get_obs``/``observation_space``. See D12 in
DECISIONS.md.

Three additions:

  obstacles     A fixed pool of 4 free-jointed bodies (``obstacle_1`` ..
                ``obstacle_4``) declared in ``assets/scene_t2.xml``. Each
                reset places ``n_obstacles`` of them on randomly chosen
                squares and parks the rest off-board on the floor. MuJoCo
                cannot add bodies after compile, hence a pool rather than a
                count.

                T2 has its OWN scene rather than sharing T1's. Adding four
                free joints to scene.xml -- parked 1.2 m away, never touched --
                moved T1's five-target sweep from 309/315 to 306/315, with 149
                of 315 episodes differing numerically: extra DOFs change the
                constraint solve, which is the sensitivity D11 records. T1's
                validated figures have to keep reproducing. scene_t2.xml is
                generated from scene.xml by scripts/build_t2_scene.py so the
                two cannot drift.

  a fifth success term
                An episode succeeds only if the obstacles were left alone, on
                top of RoboChessEnv's on_square/resting/released/settled. The
                term is applied here, by amending ``info``, so the base env's
                criterion is unchanged and ``SparsePickAndPlaceReward`` picks
                obstacles up with no code change of its own.

  obstacle observation
                The observation vector gains 4 (x, y) pairs, one per pool
                slot, so obstacle positions are visible to a policy. Parked
                slots report their parked position, so the vector is a
                constant size whatever ``n_obstacles`` is.

Reward wrappers are reused unchanged: ``ShapedPickAndPlaceReward`` and
``SparsePickAndPlaceReward`` both work on top of this wrapper. Note what that
means for the shaped arm -- its reward is a pure function of two distances and
never reads ``is_success``, so it receives **no per-step signal about
obstacles at all** and can only avoid them by accident. That asymmetry between
the two reward arms is deliberate (D12), and must be named when T2's
shaped-vs-sparse results are analysed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from pathlib import Path

from robochess.envs.robochess_env import SQUARE_PITCH, RoboChessEnv
from robochess.tasks.squares import ALL_SQUARES, FILES, RANKS, square_to_local_xy
from robochess.tasks.t1_pick_and_place import REACHABLE_SQUARES, T1PickAndPlaceTask

# The scene T2 needs: scene.xml plus the obstacle pool. Build it with
# scripts/build_t2_scene.py; RoboChessEnv takes it as `scene_path`.
T2_SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "scene_t2.xml"


# The pool declared in assets/scene_t2.xml. Changing this means changing the
# scene: MuJoCo models are static once compiled, so the number of obstacle
# bodies is fixed at model-compile time and only how many are *placed* varies
# per episode.
OBSTACLE_BODIES: tuple[str, ...] = ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")
N_OBSTACLE_SLOTS = len(OBSTACLE_BODIES)

# Where an unused slot sits: on the floor, well clear of the table, matching
# the scene's home keyframe. A parked obstacle still reports its position in
# the observation (D12) -- that is why this is a fixed, documented constant
# rather than "somewhere off to the side".
PARKED_XY: tuple[tuple[float, float], ...] = tuple(
    (1.20 + 0.20 * i, -1.20) for i in range(N_OBSTACLE_SLOTS)
)

# An obstacle counts as disturbed once its centre has moved this far in XY
# from where reset placed it. Chosen against the tolerances already in the
# env rather than invented: goal_tolerance is 30 mm, rest_tolerance 10 mm,
# the off-board margin 39 mm. 5 mm is below all of them and above the
# settling jitter of a piece that was teleported and left alone.
#
# Bare contact was rejected as the criterion. A bounding-sphere estimate put
# the open gripper 4 mm inside a file-adjacent square's envelope during
# "descend"; real-contact measurement later showed the pad actually clears
# (see file_adjacent_squares). Either way a contact test is the wrong bar --
# it fires on grazes that displace nothing, and displacement is the thing
# the task actually cares about.
OBSTACLE_DISPLACEMENT_TOL = 0.005

# ...and once it has tilted this far from upright, which is what "knocked
# over" looks like when the base barely slides. A piece lying on its side is
# ~90 deg; 15 deg is comfortably past settling noise on a teleported piece.
OBSTACLE_TILT_TOL = 0.26  # rad


def file_adjacent_squares(square: str) -> tuple[str, ...]:
    """The squares one file either side of ``square``, same rank.

    Excluded from obstacle placement as a precaution, on a conservative
    estimate rather than a confirmed clearance requirement. A sweep of the
    moving geoms against a standing piece's envelope, treating each geom as
    its MuJoCo *bounding sphere*, put ``ur_gripper_left_pad1`` ~4 mm inside
    the file-adjacent square's envelope during "descend" on every start
    square tested, with rank-adjacent squares ~14 mm clear.

    That estimate over-states the pad: bounding spheres are larger than the
    pad boxes they enclose. Checked against real contacts, an obstacle
    standing on a file-adjacent square is never touched -- across a
    315-episode sweep its only contact all episode is with the board slab,
    and the obstacle term never fired. So this exclusion is stricter than
    the physics requires. It is kept because the margin is small either way
    and because a bystander adjacent to the square being grasped is the
    placement most likely to become unsafe if the grasp or the approach
    changes; it should not be cited as a measured collision.
    """
    file_idx, rank = FILES.index(square[0]), square[1]
    return tuple(
        FILES[i] + rank for i in (file_idx - 1, file_idx + 1) if 0 <= i < len(FILES)
    )


class T2ObstacleTask(T1PickAndPlaceTask):
    """T1's pick-and-place with ``n_obstacles`` bystander pieces on the board.

    Obstacle squares are drawn at random, excluding the start and target
    squares and the file-adjacent neighbours of both (see
    :func:`file_adjacent_squares`). Pass ``obstacle_squares`` to the
    constructor, or per-episode via
    ``reset(options={"obstacle_squares": [...]})``, to fix them instead.

    Success requires everything ``RoboChessEnv`` requires *and*
    ``info["obstacles_undisturbed"]``. Disturbing an obstacle does not
    terminate the episode -- the trajectory runs to its natural end so RL
    sees a complete episode, it simply cannot be scored a success.
    """

    def __init__(
        self,
        env: RoboChessEnv,
        start_square: str | None = None,
        target_square: str | None = None,
        n_obstacles: int = 2,
        obstacle_squares: tuple[str, ...] | None = None,
        reachable_squares: tuple[str, ...] = REACHABLE_SQUARES,
        displacement_tol: float = OBSTACLE_DISPLACEMENT_TOL,
        tilt_tol: float = OBSTACLE_TILT_TOL,
    ):
        super().__init__(
            env,
            start_square=start_square,
            target_square=target_square,
            reachable_squares=reachable_squares,
        )
        if not 0 <= n_obstacles <= N_OBSTACLE_SLOTS:
            raise ValueError(
                f"n_obstacles must be in [0, {N_OBSTACLE_SLOTS}]; got {n_obstacles}."
            )
        if obstacle_squares is not None:
            if len(obstacle_squares) != n_obstacles:
                raise ValueError(
                    f"obstacle_squares has {len(obstacle_squares)} entries but "
                    f"n_obstacles is {n_obstacles}."
                )
            for sq in obstacle_squares:
                if sq not in ALL_SQUARES:
                    raise ValueError(f"Not a valid square: {sq!r}")
            if len(set(obstacle_squares)) != len(obstacle_squares):
                raise ValueError("obstacle_squares must be distinct.")

        self.n_obstacles = int(n_obstacles)
        self.obstacle_squares = tuple(obstacle_squares) if obstacle_squares else None
        self.displacement_tol = float(displacement_tol)
        self.tilt_tol = float(tilt_tol)

        base = self._base
        self._obstacle_bids = []
        self._obstacle_qadrs = []
        self._obstacle_vadrs = []
        for name in OBSTACLE_BODIES:
            bid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(
                    f"Scene has no body {name!r}. T2 needs the obstacle pool, which "
                    f"lives in assets/scene_t2.xml -- build the env with "
                    f"RoboChessEnv(scene_path=T2_SCENE_PATH), or use make_t2_env(). "
                    f"T1's scene.xml deliberately does not have it."
                )
            jid = base.model.body_jntadr[bid]
            self._obstacle_bids.append(bid)
            self._obstacle_qadrs.append(int(base.model.jnt_qposadr[jid]))
            self._obstacle_vadrs.append(int(base.model.jnt_dofadr[jid]))

        self.current_obstacle_squares: tuple[str, ...] = ()
        # Where reset put each slot, which is what displacement is measured
        # against -- not the square centre, so a placement that settles a
        # fraction of a millimetre does not read as disturbance.
        self._placed_xy = np.zeros((N_OBSTACLE_SLOTS, 2), dtype=np.float64)

        # D12: the observation extension is declared here, on the wrapper.
        # The base env's observation_space is left exactly as it is.
        base_space = self.env.observation_space
        obs_space = base_space["observation"]
        pad_low = np.full(2 * N_OBSTACLE_SLOTS, -3.0, dtype=obs_space.dtype)
        pad_high = np.full(2 * N_OBSTACLE_SLOTS, 3.0, dtype=obs_space.dtype)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=np.concatenate([obs_space.low, pad_low]),
                    high=np.concatenate([obs_space.high, pad_high]),
                    dtype=obs_space.dtype,
                ),
                "achieved_goal": base_space["achieved_goal"],
                "desired_goal": base_space["desired_goal"],
            }
        )

    # ----------------------------------------------------------- placement
    def _obstacle_candidates(self, start: str, target: str) -> list[str]:
        excluded = {start, target}
        for sq in (start, target):
            excluded.update(file_adjacent_squares(sq))
        return [s for s in self.reachable_squares if s not in excluded]

    def _resolve_obstacle_squares(
        self, start: str, target: str, requested: tuple[str, ...] | None
    ) -> tuple[str, ...]:
        if requested is not None:
            return tuple(requested)
        if self.n_obstacles == 0:
            return ()
        candidates = self._obstacle_candidates(start, target)
        if len(candidates) < self.n_obstacles:
            raise ValueError(
                f"Only {len(candidates)} squares are available for obstacles after "
                f"excluding {start}/{target} and their file-adjacent neighbours; "
                f"need {self.n_obstacles}."
            )
        idx = self._base.np_random.choice(
            len(candidates), size=self.n_obstacles, replace=False
        )
        return tuple(candidates[int(i)] for i in idx)

    def _set_obstacle_pose(self, slot: int, xy, z: float) -> None:
        base = self._base
        qadr, vadr = self._obstacle_qadrs[slot], self._obstacle_vadrs[slot]
        base.data.qpos[qadr : qadr + 2] = xy
        base.data.qpos[qadr + 2] = z
        base.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        base.data.qvel[vadr : vadr + 6] = 0.0
        self._placed_xy[slot] = xy

    def _place_obstacles(self, squares: tuple[str, ...]) -> None:
        for slot in range(N_OBSTACLE_SLOTS):
            if slot < len(squares):
                self._set_obstacle_pose(
                    slot, self._board_world_xy(squares[slot]), self._piece_rest_z
                )
            else:
                # Parked: on the floor, not on the board. z=0 puts the base
                # cylinder resting on the floor plane, so it stays put rather
                # than falling for the whole episode.
                self._set_obstacle_pose(slot, np.asarray(PARKED_XY[slot]), 0.0)

    # ------------------------------------------------------- measurements
    def _obstacle_xy(self) -> np.ndarray:
        """Current (x, y) of every pool slot, placed or parked."""
        return np.array(
            [self._base.data.xpos[bid][:2] for bid in self._obstacle_bids], dtype=np.float64
        )

    def _obstacle_tilts(self) -> np.ndarray:
        """Angle (rad) between each obstacle's own z axis and world z."""
        base = self._base
        out = np.empty(N_OBSTACLE_SLOTS, dtype=np.float64)
        for i, bid in enumerate(self._obstacle_bids):
            zz = float(base.data.xmat[bid].reshape(3, 3)[2, 2])
            out[i] = float(np.arccos(np.clip(zz, -1.0, 1.0)))
        return out

    def _obstacle_off_board(self) -> np.ndarray:
        """Per-slot: has this obstacle left the board's XY footprint?

        Same test the env applies to the tracked piece, evaluated here for the
        obstacles. Parked slots are off-board by construction, so this is only
        ever consulted for placed ones.
        """
        base = self._base
        centre = base.data.xpos[self._board_bid]
        half = base.model.geom_size[base._geom_id("board_slab")][:2]
        xy = self._obstacle_xy()
        return np.any(np.abs(xy - centre[:2]) > half + 0.5 * SQUARE_PITCH, axis=1)

    def _obstacle_status(self) -> tuple[np.ndarray, np.ndarray]:
        """``(displacement_m, disturbed)`` for the placed obstacles only."""
        n = len(self.current_obstacle_squares)
        if n == 0:
            return np.zeros(0), np.zeros(0, dtype=bool)
        moved = np.linalg.norm(self._obstacle_xy()[:n] - self._placed_xy[:n], axis=1)
        disturbed = (
            (moved > self.displacement_tol)
            | (self._obstacle_tilts()[:n] > self.tilt_tol)
            | self._obstacle_off_board()[:n]
        )
        return moved, disturbed

    # ---------------------------------------------------------------- gym
    def _extend_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Append every slot's (x, y) to the observation vector (D12)."""
        space = self.observation_space["observation"]
        extended = np.concatenate(
            [obs["observation"], self._obstacle_xy().ravel().astype(obs["observation"].dtype)]
        )
        obs = dict(obs)
        obs["observation"] = np.clip(extended, space.low, space.high).astype(space.dtype)
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = options or {}
        obs, info = super().reset(seed=seed, options=options)

        squares = self._resolve_obstacle_squares(
            self.current_start_square,
            self.current_target_square,
            options.get("obstacle_squares", self.obstacle_squares),
        )
        self.current_obstacle_squares = squares
        self._place_obstacles(squares)
        mujoco.mj_forward(self._base.model, self._base.data)
        # Re-read after mj_forward so _placed_xy is the settled pose the rest
        # of the episode is measured against.
        self._placed_xy[: len(squares)] = self._obstacle_xy()[: len(squares)]

        obs = self._base._get_obs()
        info = self._augment_info(self._base._build_info(terminated_reason=None))
        return self._extend_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return self._extend_obs(obs), reward, terminated, truncated, info

    # --------------------------------------------------------------- info
    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info = super()._augment_info(info)
        moved, disturbed = self._obstacle_status()
        undisturbed = bool(not disturbed.any())

        info["obstacle_squares"] = self.current_obstacle_squares
        info["obstacles_undisturbed"] = undisturbed
        info["obstacle_displacements_mm"] = (moved * 1000.0).tolist()
        info["max_obstacle_displacement_mm"] = float(moved.max() * 1000.0) if moved.size else 0.0
        # T1 sets this to 0 as a forward-compatible field; T2 is what it was
        # put there for.
        info["unintended_displacement_count"] = int(disturbed.sum())

        # The fifth success term, applied here so RoboChessEnv's own criterion
        # stays exactly as D3/D9 left it. SparsePickAndPlaceReward reads
        # info["is_success"], so it inherits this with no change of its own.
        base_success = bool(info["is_success"])
        info["is_success"] = bool(base_success and undisturbed)

        # The base env terminates on its OWN success criterion, which a wrapper
        # cannot influence without reaching into the env. So the fifth term can
        # block success but cannot prevent termination, and the episode would
        # otherwise end reporting terminated_reason "success" while is_success
        # is False -- the one combination that reads as a contradiction.
        # Relabel exactly that case and nothing else: every other reason
        # ("piece_off_board", "bad_contact", None) is passed through untouched.
        # scripts/scripted_expert.py copies terminated_reason into its episode
        # `outcome`, so this string is what a sweep reports for the new
        # outcome type T2 introduces.
        if base_success and not undisturbed and info.get("terminated_reason") == "success":
            info["terminated_reason"] = "success_obstacles_disturbed"
        return info


def make_t2_env(**env_kwargs) -> RoboChessEnv:
    """A ``RoboChessEnv`` on T2's scene, so callers do not have to know the path.

    Any keyword argument ``RoboChessEnv`` accepts is passed through; only
    ``scene_path`` is filled in.
    """
    env_kwargs.setdefault("scene_path", T2_SCENE_PATH)
    return RoboChessEnv(**env_kwargs)
