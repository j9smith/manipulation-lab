import logging
logger = logging.getLogger("ManipulationLab.TaskRunner")

from hydra.utils import instantiate
from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.brain.brain import Brain
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
        self.brain = self._load_brain()

        self.step_count = 0

    def _load_brain(self):
        """
        Initialises the models and the brain.
        """
        device = self.cfg.brain.device
        logger.info("Loading brain ...")
        logger.info(f"Loading model weights from {self.cfg.brain.model_weights}")

        # Load policy model with defined weights
        self.model = instantiate(self.cfg.brain.model)
        self.model.load_state_dict(
            torch.load(self.cfg.brain.model_weights, map_location=device, weights_only=True)
        )
        self.model.to(device).eval()

        # Load image encoder if defined
        if self.cfg.brain.encoder is not None:
            logger.info(f"Loading encoder weights from {self.cfg.brain.encoder_weights}")
            self.encoder = instantiate(self.cfg.brain.encoder)
            self.encoder.load_state_dict(
                torch.load(self.cfg.brain.encoder_weights, map_location=device, weights_only=True)
            )
            self.encoder.to(device).eval()
        else:
            self.encoder = None

        # Initialise brain
        return Brain(
            cfg=self.cfg,
            model=self.model,
            control_freq=self.cfg.brain.control_frequency,
            control_event=self.control_event,
            sim_dt=self.sim_dt,
            camera_keys=self.cfg.brain.camera_keys,
            proprio_keys=self.cfg.brain.proprio_keys,
            sensor_keys=self.cfg.brain.sensor_keys,
            encoder=self.encoder,
        )

    def run(self, simulation_app):
        """
        Runs the simulation loop.
        """
        self.brain.start()

        while simulation_app.is_running():
            # Step simulator and update sim time
            self.sim.step()
            self.step_count += 1
            sim_time = self.step_count * self.sim_dt
            self.brain.sim_time = sim_time

            # Inform the control loop that the sim has stepped
            self.control_event.set()
            self.control_event.clear()

            # Update buffers to reflect new sim state
            self.scene.update(self.sim_dt)

            # Get observations and push to brain
            obs = self.obs_handler.get_obs()
            self.brain.update_obs(obs)
            
            # Check for new actions from brain
            action = self.brain.get_action()
            if action is not None:
                self.action_handler.apply(action=action)