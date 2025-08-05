"""Provides an env wrapper for teleoperation"""

import logging
logger = logging.getLogger("ManipulationLab.TeleopHandler")

from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.dataset.writer import DatasetWriter
from manipulation_lab.scripts.teleop.controller_interface import ControllerInterface
import time
import torch
import random

class TeleopHandler:
    def __init__(self, env, cfg):
        self.env = env
        self.env.seed(random.randint(0, 2**32))
        self.sim = self.env.unwrapped.sim
        self.scene = self.env.unwrapped.scene
        self.robot = self.scene.articulations["robot"]
        self.sim_steps = 0
        self.sim_dt = self.sim.get_physics_dt()
        self.cfg = cfg
        self.dagger = self.cfg.dagger

        # TODO: If we're using the policy network to control the robot, we may
        # need to update ActionHandler's control_mode to reflect the action space
        if self.dagger == True:
            from threading import Event
            from manipulation_lab.scripts.control.controller import Controller

            self.control_event = Event()

            self.controller = Controller(
                cfg=self.cfg,
                control_freq=cfg.controller.control_frequency,
                sim_dt=self.sim_dt,
                control_event=self.control_event
            )

            self.controller.start()
            
        self.teleop_controller = ControllerInterface(**cfg.teleoperation.teleop_controller)
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")
        self.obs_handler = ObservationHandler(env=self.env)
        self.dataset_writer = DatasetWriter(
            env_name=self.env.env_name,
            task_name=self.env.task_name,
            task_language_instruction= self.env.task_language_instruction,
            task_phases=self.env.task_phases,
            sim_dt=self.sim.get_physics_dt(),
            dagger_mode=self.dagger,
            **cfg.dataset_writer
        )

        self.target_fps = cfg.dataset_writer.target_fps

        self._current_phase = 0
        self._last_phase_advance_step = 0

    def run_teleop(self, simulation_app):
        self._reset_scene()
        # Compute number of sim steps before capturing observations
        # Round to account for floating point precision errors in sim_dt
        capture_frequency = round(1.0 / self.target_fps / self.sim_dt)

        logger.info(
            f"Recording at {self.target_fps} FPS, which is every {capture_frequency} sim steps. "
            f"(sim_dt={self.sim_dt:.4f}s)"
            )

        if self.dagger:
            logger.info(
                f"Running in DAgger mode."
            )
        else:
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
                recorded_actions = {}
                use_policy_as_expert = False

                obs = self.obs_handler.get_obs()

                # Get the teleoperated action
                teleop_action = self.teleop_controller.get_action()

                if self.dagger:
                    # If no manual override (accounting for deadzone drift),
                    # defer to the policy as expert
                    if torch.all(torch.abs(teleop_action) < 5e-3):
                        use_policy_as_expert = True

                    self.controller.sim_step_count = self.sim_steps
                    self.controller.update_obs(obs)
                    self.control_event.set()
                    controller_action = self.controller.get_action()

                    if controller_action is not None:
                        recorded_actions["policy"] = {
                            "ee_deltas": controller_action[:6].cpu(),
                            "gripper_deltas": controller_action[6].cpu()
                        }
                    
                    if use_policy_as_expert and controller_action is not None:
                        recorded_actions["expert"] = {
                            "ee_deltas": controller_action[:6].cpu(),
                            "gripper_deltas": controller_action[6].cpu()
                        }
                    else:
                        recorded_actions["expert"] = {
                            "ee_deltas": teleop_action[:6].cpu(),
                            "gripper_deltas": teleop_action[6].cpu()
                        }
                else:
                    recorded_actions["expert"] = {
                            "ee_deltas": teleop_action[:6].cpu(),
                            "gripper_deltas": teleop_action[6].cpu()
                        }

                # Record observations at target FPS
                if self.sim_steps % capture_frequency == 0:
                    if self.dagger and use_policy_as_expert == False:
                        logger.info(
                            "Teleop override. Logging expert demonstration."
                        )
                    is_first = (self.sim_steps == 0)
                    task_complete, _ = self.env.get_dones()
                    is_last = task_complete.cpu()
                    self.dataset_writer.append_frame(
                        obs=obs,
                        actions=recorded_actions,
                        task_phase=self._current_phase,
                        is_first=is_first,
                        is_last=is_last,
                        sim_steps=self.sim_steps
                    )

                if self.dagger:
                    if controller_action is not None:
                        self.action_handler.apply(action=controller_action)
                else:
                    self.action_handler.apply(action=teleop_action)

                self.sim_steps += 1

            self.sim.step()
            self.scene.update(self.sim_dt)
        
    def _reset_scene(self):
        if self.dagger:
            self.controller.reset()

        # TODO: Do some ablations here to find out what we can get rid of
        self.env.reset()
        self.sim.step()

        self.scene.update(self.sim_dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos)
        self.scene.write_data_to_sim()

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