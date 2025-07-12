"""Provides an env wrapper for teleoperation"""

import logging
logger = logging.getLogger("ManipulationLab.TeleopHandler")

from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.dataset.writer import DatasetWriter
from manipulation_lab.scripts.teleop.controller_interface import ControllerInterface
from isaacsim.core.utils.stage import create_new_stage
import time

class TeleopHandler:
    def __init__(self, env, cfg):
        self.env = env
        self.sim = self.env.unwrapped.sim
        self.scene = self.env.unwrapped.scene
        self.sim_steps = 0
        self.sim_dt = self.sim.get_physics_dt()
        self.cfg = cfg

        self.teleop_controller = ControllerInterface(**cfg.teleop_controller)
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")
        self.obs_handler = ObservationHandler(env=self.env)
        self.dataset_writer = DatasetWriter(
            env_name=self.env.env_name,
            task_name=self.env.task_name,
            sim_dt=self.sim.get_physics_dt(),
            **cfg.dataset_writer
        )

        self.target_fps = cfg.dataset_writer.target_fps

    def run_teleop(self, simulation_app):

        sim_dt = self.sim.get_physics_dt()

        self.target_fps = 30

        # Compute number of sim steps before capturing observations
        # Round to account for floating point precision errors in sim_dt
        capture_frequency = round(1.0 / self.target_fps / sim_dt)

        logger.info(
            f"Recording at {self.target_fps} FPS, which is every {capture_frequency} sim steps. "
            f"(sim_dt={sim_dt:.4f}s)"
            )

        while simulation_app.is_running():
            episode_command = self.teleop_controller.get_episode_commands()
            if episode_command is None and self.dataset_writer.episode_started:
                _continue_episode()
            else:
                if episode_command == "start":
                    self.dataset_writer.start_episode()
                    _continue_episode()
                elif episode_command == "pause":
                    self.dataset_writer.pause_episode()
                    time.sleep(0.5)
                elif episode_command == "abort":
                    self.dataset_writer.abort_episode()
                    time.sleep(0.5)
                    _reset_scene()
                elif episode_command == "finish":
                    self.dataset_writer.end_episode()
                    time.sleep(0.5)
                    _reset_scene()

            def _reset_scene():
                logger.info("Resetting scene ...")
                # Close the environment and reset the stage
                # TODO: Unsure if env.close() does anything, maybe remove?
                self.env.close()
                create_new_stage()

                # TODO: Pull this logic out into a separate helper function
                # We use the same logic in play.py, and will probably need it elsewhere
                from hydra.utils import instantiate
                self.env = instantiate(self.cfg.task)
                self.sim = self.env.unwrapped.sim
                self.scene = self.env.unwrapped.scene

                settle_steps = int(1.5 / self.sim_dt)
                for _ in range(settle_steps):
                    self.sim.step()
                    self.scene.update(self.sim_dt)
                ########################################

                # Reinitialise the action and observation handlers with new environment
                self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")
                self.obs_handler = ObservationHandler(env=self.env)

                logger.info("Scene reset.")

            def _continue_episode():
                self.sim_steps += 1

                # Get the teleoperated action
                action = self.teleop_controller.get_action()

                # Record observations at target FPS
                if self.sim_steps % capture_frequency == 0:
                    is_first = (self.sim_steps == 0)
                    is_last = False # TODO: Add _get_dones to task scene, then put it here (returns bool)
                    obs = self.obs_handler.get_obs()
                    self.dataset_writer.append_frame(
                        obs=obs,
                        action={"ee_deltas": action[:6], "gripper_deltas": action[6]},
                        is_first=is_first,
                        is_last=is_last,
                        sim_steps=self.sim_steps
                    )

                # Apply the teleoperated action to the robot
                self.action_handler.apply(action=action)

            # Step the simulation
            self.sim.step()

            # Update buffers to reflect new sim state
            self.scene.update(sim_dt)
        