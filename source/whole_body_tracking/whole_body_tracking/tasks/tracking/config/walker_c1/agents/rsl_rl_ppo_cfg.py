from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkerC1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "walker_c1_fix"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_hidden_dims=[768, 384, 192],
        critic_hidden_dims=[768, 384, 192],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    noise_std_schedule: dict = {
        "init_std": 0.35,
        "final_std": 0.08,
        "decay_iters": 1500,
        "mode": "exp",
        "entropy_init": 0.012,
        "entropy_final": 0.006,
    }
