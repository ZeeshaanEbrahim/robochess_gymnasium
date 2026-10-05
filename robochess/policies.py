"""Custom SB3 policy classes (D29). Not imported at package scope: train.py's
`build_model()` imports this module only when a config asks for
`ppo_mean_squash`, and `scripts/eval_probe.py` imports it unconditionally at
module scope so a saved model built from it can always be reloaded (SB3
pickles the policy class by reference -- see DECISIONS.md D29).
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class SquashedMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    """D29 ppo_mean_squash: tanh the Gaussian mean into [-1, 1] before it
    becomes the distribution's mean, everywhere a distribution is built
    (`forward`, `predict`, `get_distribution`, `evaluate_actions` all call
    `_get_action_dist_from_latent` -- this is the only override needed for
    training and eval to agree). Noise (`log_std`), action clipping and the
    log-prob formula are otherwise unchanged; only the mean moves.
    """

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if isinstance(self.action_dist, DiagGaussianDistribution):
            mean_actions = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(th.tanh(mean_actions), self.log_std)
        return super()._get_action_dist_from_latent(latent_pi)
