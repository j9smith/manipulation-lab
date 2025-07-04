import gymnasium as gym

import manipulation_lab.envs.tabletop.tasks.blocks_cfg

gym.register(
    id="Isaac-Blocks-v0",
    entry_point="isaaclab.envs:DirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "manipulation_lab.envs.tabletop.tasks.blocks_cfg:BlocksEnvCfg"
    }
)

print("Tasks registered.")