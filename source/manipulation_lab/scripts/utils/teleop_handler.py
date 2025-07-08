"""Provides an env wrapper for teleoperation"""

import logging
logger = logging.getLogger("ManipulationLab.TeleopHandler")

from manipulation_lab.scripts.utils.action_handler import ActionHandler
from multiprocessing import Array
import torch

class TeleopHandler:
    def __init__(self, env, remote_connection=True):
        self.env = env
        self.sim = self.env.unwrapped.sim
        self.scene = self.env.unwrapped.scene
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")
        self.remote_connection = remote_connection
        self.action_array: Array = None
        
        if remote_connection:
            self._initialise_socket_connection()

    def _initialise_socket_connection(self):
        """
        Initialise a socket connection to accept teleoperation from remote server
        """
        from manipulation_lab.scripts.utils.socket_listener import start_socket_listener
        logger.info(f"Initialising remote connection")

        # Initialise a shared array between the main process and the socket listener (threaded)
        self.action_array = Array('f', [0.0] * 7)
        self.thread = start_socket_listener(self.action_array)

    def _get_action(self):
        if self.remote_connection and self.action_array is not None:
            action = list(self.action_array)
        else: 
            action = [0.0] * 7

        return torch.tensor(action, dtype=torch.float32)

    def run_teleop(self, simulation_app):
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")

        sim_dt = self.sim.get_physics_dt()

        target_fps = 30

        # Compute number of sim steps before capturing observations
        # Round to account for floating point precision errors in sim_dt
        capture_frequency = round(1.0 / target_fps / sim_dt)

        logger.info(f"Recording at {target_fps} FPS, which is every {capture_frequency} sim steps. (sim_dt={sim_dt:.4f}s)")

        sim_steps = 0

        while simulation_app.is_running():
            sim_steps += 1

            # Record observations at target FPS
            if sim_steps % capture_frequency == 0:
                pass

            # Get the teleoperation action
            action = self._get_action()

            # Apply the teleoperation action to the robot
            self.action_handler.apply(action=action)

            # Step the simulation
            self.sim.step()

            # Update buffers to reflect new sim state
            self.scene.update(sim_dt)
        