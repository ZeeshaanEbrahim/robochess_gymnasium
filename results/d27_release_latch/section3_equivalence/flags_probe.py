import numpy as np, robochess
from robochess.envs.robochess_env import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask
print("code:", robochess.__file__)
for mode in ("absolute", "delta", "cartesian"):
    base = RoboChessEnv(action_mode=mode)
    env = T1PickAndPlaceTask(base)
    env.reset(seed=0)
    u = env.unwrapped
    at_reset = env._piece_contacts()
    hold = u.home_action() if mode == "absolute" else np.zeros(u.action_space.shape, dtype=np.float32)
    env.step(hold)
    print(mode, "reset", at_reset, "after first hold step", env._piece_contacts())
