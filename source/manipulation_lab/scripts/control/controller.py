import logging
logger = logging.getLogger("ManipulationLab.Controller")

import threading
from typing import List, Optional
from time import perf_counter
import torch

import manipulation_lab.scripts.control.model_handler as model_handler

class Controller:
    def __init__(
        self,
        cfg,
        control_freq: int,
        sim_dt: float,
        control_event: threading.Event,
    ):
        """
        # TODO: Write Controller init docstring
        """
        self.model_handler = model_handler.ModelHandler(
            cfg=cfg,
        )
        self.cfg = cfg
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq
        self.sim_dt = sim_dt

        self._shared_obs = {}
        self._shared_obs_lock = threading.Lock()

        self._action_buffer = []
        self._action_buffer_lock = threading.Lock()
        self._last_action = None
        self._action_schedule = []

        self.control_event = control_event

        self.sim_time = 0.0
        self.sim_step_count = 0
        self._last_control_step = 0
        self._next_control_step = 0 
        self._steps_per_control = int(round(1.0/(self.sim_dt * self.control_freq)))
        self._running = False
        self._thread = None

        sim_freq = 1.0 / self.sim_dt
        ratio = sim_freq / control_freq
        if ratio % 1 != 0:
            drift_per_step = (ratio % 1) * self.sim_dt
            logger.warning(
                f"Sim frequency ({sim_freq:.0f}Hz) is not a multiple of control frequency ({control_freq}Hz). "
                f"This will cause drift of {(drift_per_step * control_freq):.4f}s per second, "
                f"resulting in a missed control step every {1/(drift_per_step * control_freq):.1f} seconds. "
                f"Adjust either to avoid instability."
            )

    def update_obs(self, obs: dict):
        """
        Updates the shared observation dictionary with the latest observation.

        Called by the simulator on each sim step.
        """
        with self._shared_obs_lock:
            self._shared_obs = obs

    def get_action(self):
        """
        Returns the next action from the action buffer if it exists.

        Called by the simulator on each sim step.
        """
        with self._action_buffer_lock:
            if len(self._action_buffer) > 0:
                if self._action_buffer[0][0] <= self.sim_step_count:
                    _, action = self._action_buffer.pop(0)
                    self._last_action = action
                    return action
                else: return None
            else:
                # TODO: If we're trained on ManipLab dataset we need to consider that
                # actions may need to be applied consistently across control steps
                # e.g., if trained on 30Hz, we need to apply as if we operated at 60Hz
                return None 

    def _get_latest_obs(self):
        """
        Returns the latest observation from the shared observation dictionary.

        Called by the controller on each control step.
        """
        with self._shared_obs_lock:
            return self._shared_obs

    def _push_actions(self, actions):
        """
        Stores actions in the action buffer. Overwrites any stale actions.

        Called by the controller on each control step.

        TODO: Rewrite to maintain action across multiple steps while supporting action chunking.
        """
        self._schedule_actions(actions)

    def _schedule_actions(self, actions):
        """
        Schedules actions across the control step. Allows multiple actions to be scheduled
        across a single control step (e.g., for action chunking transformers).
        """
        actions = actions.unsqueeze(0) if actions.ndim == 1 else actions
        chunk_size = actions.shape[0]

        # The number of sim steps per control step
        sim_steps_per_control_step = self._steps_per_control
        
        # The number of sim steps to wait between actions
        sim_steps_per_action = int(sim_steps_per_control_step / chunk_size)

        # The sim step of the first action (i.e., on the next control step)
        action_step = self._next_control_step

        # Schedule actions evenly across the control step
        with self._action_buffer_lock:
            self._action_buffer.clear()
            for i in range(chunk_size):
                self._action_buffer.append((action_step, actions[i]))
                action_step += sim_steps_per_action
        
    def _step(self):
        """
        Performs a single control step. 
        """
        # TODO: What about unstructured data?
        inference_start_time = perf_counter()

        raw_obs = self._get_latest_obs()
        actions = self.model_handler.forward(raw_obs)

        inference_latency = perf_counter() - inference_start_time
        logger.debug(f"Inference latency: {inference_latency*100:.2f}ms")

        if inference_latency > self.control_dt:
            logger.warning(
                f"Inference latency ({inference_latency*1000:.2f}ms) exceeded "
                f"control step time ({self.control_dt*1000:.2f}ms)."
            )

        # Send actions (tensor) to the action buffer
        self._push_actions(actions)

    def run(self):
        """
        Runs the control loop. This exists on a separate thread to enforce separation of
        simulation and robot control to imitate real-world deployment.

        The control loop abides by simulation time.
        """
        self._running = True

        drift_step_tolerance = 1

        while self._running:
            # Wait for the sim to step before checking for control step
            self.control_event.wait()
            self.control_event.clear()

            if self.sim_step_count >= self._next_control_step:
                if(self.sim_step_count > self._next_control_step + drift_step_tolerance):
                    logger.warning(
                        f"Sim step {self.sim_step_count} is out of sync with control step "
                        f"{self._next_control_step} by {self.sim_step_count - self._next_control_step} steps"
                    )
                    self._next_control_step = self.sim_step_count + self._steps_per_control
                else: self._next_control_step += self._steps_per_control
                self._step()

    def start(self):
        """
        Starts the control loop asynchronously from the simulation. This is to imitate
        real-world deployment.
        """
        logger.info(f"Starting control loop at {self.control_freq}Hz")
        self._next_control_step = self.sim_step_count + self._steps_per_control
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stops the control loop.
        """
        self._running = False
        if self._thread:
            self._thread.join()
