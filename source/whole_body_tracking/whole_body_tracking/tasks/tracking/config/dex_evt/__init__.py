import gymnasium as gym

from . import agents, simple_env_cfg

##
# Register Gym environments for Dex EVT.
##

gym.register(
    id="Tracking-Flat-DexEVT-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": simple_env_cfg.DexEVTSimpleEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)
