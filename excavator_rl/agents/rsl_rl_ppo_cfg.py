"""PPO configuration for rsl_rl.

vortexRL found PPO and DDPG to converge fastest of the four algorithms it
compared (REINFORCE / DDPG / PPO / TRPO), with PPO the more stable of the two
-- so PPO is what this port uses, in its massively-parallel Isaac Lab form.
Their single-environment Vortex setup needed ~1000 episodes / <10 h on an
RTX 2060; with 4096 parallel environments the same experience arrives in
minutes.

The config works with both rsl_rl generations shipped by Isaac Lab:

* Isaac Lab 2.2 / rsl-rl-lib 2.3.x: observation normalisation is the runner
  level ``empirical_normalization`` flag.
* Isaac Lab 2.3 / rsl-rl-lib 3.x: normalisation moved into the policy
  (``actor_obs_normalization`` / ``critic_obs_normalization``) and the runner
  needs ``obs_groups``.  Leaving those unset crashes rsl_rl 3.x.
"""

from isaaclab.utils import configclass

try:  # Isaac Lab >= 2.0
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )
except ImportError:  # pragma: no cover - Isaac Lab 1.x layout
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # type: ignore
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )

#: True for Isaac Lab >= 2.3 (rsl-rl-lib >= 3.0)
_NEW_RSL_RL_API = "actor_obs_normalization" in getattr(RslRlPpoActorCriticCfg, "__dataclass_fields__", {})

#: running mean/std normalisation of the observations (tip xyz is in metres,
#: the rest is roughly unit scale)
OBS_NORMALIZATION = True

_policy_kwargs = dict(
    init_noise_std=0.8,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)
if _NEW_RSL_RL_API:
    _policy_kwargs.update(
        actor_obs_normalization=OBS_NORMALIZATION,
        critic_obs_normalization=OBS_NORMALIZATION,
    )


@configclass
class ExcavatorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 32
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "excavator_digging"

    if _NEW_RSL_RL_API:
        # the env returns a single "policy" observation group; the critic sees
        # the same observations (no privileged state)
        obs_groups: dict = {"policy": ["policy"], "critic": ["policy"]}
    else:
        empirical_normalization: bool = OBS_NORMALIZATION

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(**_policy_kwargs)

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
