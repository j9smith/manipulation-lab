from manipulation_lab.scripts.utils.action_handler import ActionHandler
from manipulation_lab.scripts.utils.obs_handler import ObservationHandler

import gymnasium as gym
from typing import Any

import torch

from isaaclab.envs.direct_rl_env import DirectRLEnv

class TaskRunner:
    """
    Build and execute task logic
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.scene = None
        self.sim = None
        self.obs_handler = ObservationHandler
        self.action_handler = ActionHandler

    def run(self, simulation_app, env_cfg, task):
        env = DirectRLEnv(cfg=env_cfg)

        self.sim = env.unwrapped.sim
        self.scene = env.unwrapped.scene

        self.obs_handler = ObservationHandler()

        # Allow the simulation to warm up
        settle_steps = int(5.0 / self.sim.get_physics_dt())
        for _ in range(settle_steps):
            self.sim.step()
            sim_dt = self.sim.get_physics_dt()
            self.scene.update(sim_dt)

        self.action_handler = ActionHandler(env=env, control_mode="delta_cartesian")

        while simulation_app.is_running():
            # Step the simulation
            self.sim.step()

            # Update buffers to reflect new sim state
            sim_dt = self.sim.get_physics_dt()
            self.scene.update(sim_dt)

            self.obs_handler.get_obs(env)
            action_tensor = torch.tensor([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], device="cuda")
            self.action_handler.apply(action=action_tensor)