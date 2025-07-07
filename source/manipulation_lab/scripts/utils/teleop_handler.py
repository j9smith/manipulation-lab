"""Provides an env wrapper for teleoperation"""

from manipulation_lab.scripts.utils.action_handler import ActionHandler
from multiprocessing import Array
import gymnasium as gym
from typing import Any
import torch
from isaaclab.envs.direct_rl_env import DirectRLEnv

class TeleopHandler:
    def __init__(self, remote_connection=True):
        self.action_handler = ActionHandler
        self.remote_connection = remote_connection
        self.action_array: Array = None
        
        if remote_connection:
            self._initialise_socket_connection()

    def _initialise_socket_connection(self):
        from manipulation_lab.scripts.utils.socket_listener import start_socket_listener

        # Initialise a shared array between the main process and the socket listener (threaded)
        self.action_array = Array('f', [0.0] * 6)
        self.thread = start_socket_listener(self.action_array)

    def _get_action(self):
        if self.remote_connection and self.action_array is not None:
            action = list(self.action_array)
        else: 
            action = [0.0] * 6

        return torch.tensor(action, dtype=torch.float32)

    def run_teleop(self, simulation_app, env_cfg, task):
        env = DirectRLEnv(cfg=env_cfg)

        self.sim = env.unwrapped.sim
        self.scene = env.unwrapped.scene

        # Allow the simulation to warm up
        settle_steps = int(5.0 / self.sim.get_physics_dt())
        for _ in range(settle_steps):
            self.sim.step()
            sim_dt = self.sim.get_physics_dt()
            self.scene.update(sim_dt)

        self.action_handler = ActionHandler(env=env, control_mode="delta_cartesian")

        while simulation_app.is_running():
            # Step the simulation
            action = self._get_action()
            print(f"Action: {action}")

            # Update buffers to reflect new sim state
            sim_dt = self.sim.get_physics_dt()
            self.scene.update(sim_dt)

            self.action_handler.apply(action=action)
            self.sim.step()
        