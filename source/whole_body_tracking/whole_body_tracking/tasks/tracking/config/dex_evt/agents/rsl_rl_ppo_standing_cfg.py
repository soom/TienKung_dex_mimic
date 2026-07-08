"""PPO runner config for Dex EVT standing-only training.

Inherits teleop teacher's network architecture [768, 384, 192] but uses
ultra-low initial noise — standing requires near-zero action from the start.

Training signal is pure tracking-style rewards: match the standing reference
(joint pose, zero velocity, standing FK).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass
class DexEVTStandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000    # was 5000 — more time for precision convergence
    save_interval = 200
    experiment_name = "dex_evt_standing"
    empirical_normalization = True

    policy_class_name = "ActorCritic"
    policy_kwargs = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [768, 384, 192],   # match teleop teacher
        "critic_hidden_dims": [768, 384, 192],   # match teleop teacher
        "activation": "elu",
        "init_noise_std": 0.05,   # ultra-low: PD holds standing pose initially
    }

    policy = dict(policy_kwargs)

    algorithm = {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 5e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    }

    noise_std_schedule: dict = {
        "init_std": 0.20,
        "final_std": 0.02,    # was 0.03 — tighter convergence
        "decay_iters": 3000,  # was 1000 — slower decay, more exploration time
        "mode": "exp",
        "entropy_init": 0.005,
        "entropy_final": 0.001,
    }
