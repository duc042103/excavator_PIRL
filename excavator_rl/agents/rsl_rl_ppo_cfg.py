"""PPO configuration for rsl_rl.

vortexRL found PPO and DDPG to converge fastest of the four algorithms it
compared (REINFORCE / DDPG / PPO / TRPO), with PPO the more stable of the two
-- so PPO is what this port uses, in its massively-parallel Isaac Lab form.
Their single-environment Vortex setup needed ~1000 episodes / <10 h on an
RTX 2060; with 4096 parallel environments the same experience arrives in
minutes.
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


@configclass
class ExcavatorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 32
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "excavator_digging"
    empirical_normalization: bool = True

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

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
