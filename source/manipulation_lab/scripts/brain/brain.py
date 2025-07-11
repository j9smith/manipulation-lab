import logging
logger = logging.getLogger("ManipulationLab.Brain")

import threading
from typing import List, Optional
from time import perf_counter
import torch

class Brain:
    def __init__(
        self,
        cfg,
        model,
        control_freq: int,
        sim_dt: float,
        control_event: threading.Event,
        camera_keys: Optional[List[str]] = None,
        proprio_keys: Optional[List[str]] = None,
        sensor_keys: Optional[List[str]] = None,
        encoder: Optional[torch.nn.Module] = None,
    ):
        """
        # TODO: Write Brain init docstring
        """
        self.cfg = cfg
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq
        self.sim_dt = sim_dt

        self.model = model
        self.encoder = encoder

        self.camera_keys = camera_keys
        self.proprio_keys = proprio_keys
        self.sensor_keys = sensor_keys

        self._shared_obs = {}
        self._shared_obs_lock = threading.Lock()

        self._action_buffer = []
        self._action_buffer_lock = threading.Lock()

        self.control_event = control_event

        self.sim_time = 0.0
        self._step_count = 0
        self._last_control_time = 0.0
        self._next_control_time = 1.0 # Allow sim to settle before first control step
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
                return self._action_buffer.pop(0)
            else: return None

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
        """
        with self._action_buffer_lock:
            self._action_buffer = [actions]

    def _schedule_actions(self, actions):
        """
        Schedules actions across the control step. Allows multiple actions to be scheduled
        across a single control step (e.g., for action chunking transformers).
        """
        pass

    def _extract_desired_obs(self, obs: dict):
        """
        Extracts the desired observation from the raw observation dictionary.
        """
        # TODO: What do we do if images are different dims? What if we want sensor and proprio data too?
        def _get_nested_data(nested_dict: dict, key: str):
            keys = key.split("/")
            for k in keys:
                nested_dict = nested_dict[k]
            return nested_dict

        camera_obs = {}
        if self.camera_keys:
            for key in self.camera_keys:
                camera_obs[key] = _get_nested_data(obs, key)

        proprio_obs = {}
        if self.proprio_keys:
            for key in self.proprio_keys:
                proprio_obs[key] = _get_nested_data(obs, key)

        sensor_obs = {}
        if self.sensor_keys:
            for key in self.sensor_keys:
                sensor_obs[key] = _get_nested_data(obs, key)

        return camera_obs, proprio_obs, sensor_obs
        
    def _step(self):
        """
        Performs a single control step. 
        """
        # TODO: What about unstructured data?
        inference_start_time = perf_counter()

        # Get the latest observation and extract target data
        raw_obs = self._get_latest_obs()
        camera_obs, proprio_obs, sensor_obs = self._extract_desired_obs(raw_obs)

        obs = []

        # Process all target data and append to obs list
        if self.encoder is not None:
            for _, value in camera_obs.items():
                cam_latent_obs = self.encoder(value)
                obs.append(cam_latent_obs)

        if self.proprio_keys:
            for _, value in proprio_obs.items():
                value = torch.tensor(value)
                obs.append(value)

        if self.sensor_keys:
            # TODO: How do we handle multidimensional sensor data?
            for _, value in sensor_obs.items():
                value = torch.tensor(value)
                obs.append(value)

        # Concatenate all target data into flat tensor
        obs = torch.cat(obs, dim=-1).to(self.cfg.brain.device)

        # Run the model
        with torch.no_grad():
            actions = self.model(obs)

        inference_latency = perf_counter() - inference_start_time
        logger.debug(f"Inference latency: {inference_latency*100:.2f}ms")

        # Send actions (tensor) to the action buffer
        self._push_actions(actions)

    def run(self):
        """
        Runs the control loop. This exists on a separate thread to enforce separation of
        simulation and robot control to imitate real-world deployment.

        The control loop abides by simulation time.
        """
        self._running = True

        drift_tolerance = self.sim_dt * 1.5
        logger.debug(f"Drift tolerance: {drift_tolerance}s")

        while self._running:
            # Wait for the sim to step before checking for control step
            self.control_event.wait(timeout=self.control_dt)

            if self.sim_time >= self._next_control_time:
                if(self.sim_time - self._next_control_time > drift_tolerance):
                    logger.warning(
                        f"Sim time {self.sim_time:.4f} is out of sync with control time "
                        f"{self._next_control_time:.4f} by {self.sim_time - self._next_control_time:.4f}s"
                    )
                self._step()
                self._next_control_time += self.control_dt

    def start(self):
        """
        Starts the control loop asynchronously from the simulation. This is to imitate
        real-world deployment.
        """
        logger.info(f"Starting control loop at {self.control_freq}Hz")
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stops the control loop.
        """
        self._running = False
        if self._thread:
            self._thread.join()
