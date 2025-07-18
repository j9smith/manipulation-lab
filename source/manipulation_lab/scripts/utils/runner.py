import logging
logger = logging.getLogger("ManipulationLab.TaskRunner")

from hydra.utils import instantiate
from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.control.controller import Controller
import torch

from threading import Event

class TaskRunner:
    """
    Build and execute task logic
    """
    def __init__(self, cfg, env):
        # Environment variables
        self.cfg = cfg
        self.env = env
        self.scene = self.env.unwrapped.scene
        self.sim = self.env.unwrapped.sim
        self.sim_dt = self.sim.get_physics_dt()
        self.control_event = Event()

        # Control handlers
        self.obs_handler = ObservationHandler(env=self.env)
        self.action_handler = ActionHandler(env=self.env, control_mode="delta_cartesian")

        # Models
        self.model = None
        self.encoder = None
        self.controller = self._load_controller()

        self.step_count = 0

    def _load_controller(self):
        """
        Initialises the models and the controller.
        """
        return Controller(
            cfg=self.cfg,
            control_freq=self.cfg.controller.control_frequency,
            control_event=self.control_event,
            sim_dt=self.sim_dt,
        )

    def run(self, simulation_app):
        """
        Runs the simulation loop.
        """
        self.controller.start()

        while simulation_app.is_running():
            # Step simulator and update sim time
            self.sim.step()
            self.step_count += 1
            sim_time = self.step_count * self.sim_dt
            self.controller.sim_time = sim_time
            self.controller.sim_step_count = self.step_count

            # Get observations and push to controller
            obs = self.obs_handler.get_obs()
            self.controller.update_obs(obs)

            # Inform the control loop that the sim has stepped
            self.control_event.set()

            # Update buffers to reflect new sim state
            self.scene.update(self.sim_dt)
            
            # Check for new actions from controller
            action = self.controller.get_action()
            if action is not None:
                self.action_handler.apply(action=action)