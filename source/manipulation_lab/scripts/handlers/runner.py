from manipulation_lab.scripts.handlers.action_handler import ActionHandler
from manipulation_lab.scripts.handlers.obs_handler import ObservationHandler

import gymnasium as gym

# import gymnasium registers to make tasks visible
import manipulation_lab.envs.tabletop.tasks.tabletop_register

class TaskRunner:
    """Build and execute task logic"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.scene = None
        self.sim = None

    def run(self, simulation_app, env_cfg, task):
        env = gym.make(task, cfg=env_cfg)

        sim = env.sim

        while simulation_app.is_running():
            sim.step()