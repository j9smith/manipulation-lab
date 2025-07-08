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
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.scene = self.env.unwrapped.scene
        self.sim = self.env.unwrapped.sim
        self.obs_handler = ObservationHandler(env=self.env)
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")

    def run(self, simulation_app):
        while simulation_app.is_running():
            # Step the simulation
            self.sim.step()

            # Update buffers to reflect new sim state
            sim_dt = self.sim.get_physics_dt()
            self.scene.update(sim_dt)

            self.obs_handler.get_obs()
            action_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda")
            self.action_handler.apply(action=action_tensor)
            break