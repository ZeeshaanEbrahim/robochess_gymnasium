"""RoboChessGymnasium: a UR5e + Robotiq 2F-85 chess-piece manipulation benchmark."""

from gymnasium.envs.registration import register, registry

__all__ = ["ENV_ID"]

ENV_ID = "RoboChess-v0"

# Registered so the env can be built with gymnasium.make(). This also gives the
# env a `spec`, which gymnasium's check_env needs in order to exercise the
# alternative render modes (without one it warns and skips that check).
#
# Deliberately no `max_episode_steps=` here: the env truncates itself against
# its own `max_episode_steps` argument, and adding a TimeLimit wrapper on top
# would truncate twice, at whichever limit is smaller.
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="robochess.envs.robochess_env:RoboChessEnv",
    )
