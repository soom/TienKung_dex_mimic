from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexEVTFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "dex_evt_fix"
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
        entropy_coef=0.012,    # 初始高探索，由 noise_std_schedule 衰减到 0.004
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # Noise curriculum: 前期高噪声快速探索，后期收敛
    # std: 0.35 → 0.05 (exp decay over 8000 iters)
    # entropy_coef: 0.012 → 0.004 同步衰减
    noise_std_schedule: dict = {
        "init_std": 0.35,
        "final_std": 0.05,
        "decay_iters": 8000,
        "mode": "exp",
        "entropy_init": 0.012,
        "entropy_final": 0.004,
    }
