"""Provides an env wrapper for teleoperation"""

import logging
logger = logging.getLogger("ManipulationLab.TeleopHandler")

from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.dataset.writer import DatasetWriter
from manipulation_lab.scripts.teleop.controller_interface import ControllerInterface
import time

class TeleopHandler:
    def __init__(self, env, cfg):
        self.env = env
        self.sim = self.env.unwrapped.sim
        self.scene = self.env.unwrapped.scene
        self.robot = self.scene.articulations["robot"]
        self.sim_steps = 0
        self.sim_dt = self.sim.get_physics_dt()
        self.cfg = cfg

        self.teleop_controller = ControllerInterface(**cfg.teleoperation.teleop_controller)
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")
        self.obs_handler = ObservationHandler(env=self.env)
        self.dataset_writer = DatasetWriter(
            env_name=self.env.env_name,
            task_name=self.env.task_name,
            task_language_instruction= self.env.task_language_instruction,
            task_phases=self.env.task_phases,
            sim_dt=self.sim.get_physics_dt(),
            **cfg.dataset_writer
        )

        self.target_fps = cfg.dataset_writer.target_fps

        self._current_phase = 0
        self._last_phase_advance_step = 0

    def run_teleop(self, simulation_app):
        # Compute number of sim steps before capturing observations
        # Round to account for floating point precision errors in sim_dt
        capture_frequency = round(1.0 / self.target_fps / self.sim_dt)

        logger.info(
            f"Recording at {self.target_fps} FPS, which is every {capture_frequency} sim steps. "
            f"(sim_dt={self.sim_dt:.4f}s)"
            )

        task_phases = []
        for i in range(len(self.env.task_phases)):
            task_phases.append(
                f"Phase {i}: {self.env.task_phases[i]}\n"
            )
        logger.info(f"Task Phases:\n" + "".join(task_phases))
        logger.info(f"Current phase: [{self._current_phase}] {self.env.task_phases[self._current_phase]}")

        while simulation_app.is_running():
            episode_command = self.teleop_controller.get_episode_commands()
            if episode_command is None and self.dataset_writer.episode_started:
                _continue_episode()
            else:
                if episode_command == "start":
                    self.dataset_writer.start_episode()
                    self.sim_steps = 0
                    self._current_phase = 0
                    _continue_episode()
                elif episode_command == "pause":
                    # TODO: Retired this for now in favour of advancing phase
                    self.dataset_writer.pause_episode()
                    time.sleep(0.5)
                elif episode_command == "abort":
                    self.dataset_writer.abort_episode()
                    time.sleep(0.5)
                    self._reset_scene()
                elif episode_command == "finish":
                    self.dataset_writer.end_episode()
                    time.sleep(0.5)
                    self._reset_scene()
                elif episode_command == "advance_phase":
                    self._advance_phase()

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
                        task_phase=self._current_phase,
                        is_first=is_first,
                        is_last=is_last,
                        sim_steps=self.sim_steps
                    )

                # Apply the teleoperated action to the robot
                self.action_handler.apply(action=action)

            # Step the simulation
            self.sim.step()

            # Update buffers to reflect new sim state
            self.scene.update(self.sim_dt)
        
    def _reset_scene(self):
        # TODO: Do some ablations here to find out what we can get rid of
        logger.info("Resetting scene ...")
        self.env.reset()
        self.sim.step()

        self.scene.update(self.sim_dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos)
        self.scene.write_data_to_sim()

        settle_steps = int(0.5 / self.sim_dt)
        for _ in range(settle_steps):
            self.sim.step()
            self.scene.update(self.sim_dt)
            
        logger.info("Scene reset.")
    
    def _advance_phase(self):
        # Avoid repeat commands without interrupting sim
        if self.sim_steps >= self._last_phase_advance_step + 10: 
            if self._current_phase < len(self.env.task_phases) - 1:
                self._current_phase += 1
                logger.info(
                    f"Advancing to phase {self._current_phase}: "
                    f"{self.env.task_phases[self._current_phase]}"
                    )
                self._last_phase_advance_step = self.sim_steps
            else: logger.info("Already in terminal phase. Cannot advance.")