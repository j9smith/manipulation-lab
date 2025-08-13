# Config imports
from isaaclab.utils import configclass
from omegaconf import MISSING

# Env imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg
import isaaclab.envs.mdp.events as events

# Scene imports
from isaaclab.scene import InteractiveSceneCfg

# Asset imports
from isaaclab.assets.articulation import ArticulationCfg

import manipulation_lab.envs.utils as utils
import torch

@configclass
class SceneCfg(InteractiveSceneCfg):
    """
    Design the scene
    """
    robot: ArticulationCfg = MISSING

@configclass
class RandomEventCfg:
    """
    Scene randomisation logic goes here.
    """
    pass

@configclass
class ResetEnvCfg:
    """
    Deterministically reset the environment.
    """
    deterministic_env_reset = EventTermCfg(
        func=utils.reset_env,
        mode="reset",
        params={},
    )

@configclass
class EnvCfg(DirectRLEnvCfg):
    """
    Create a config for the environment
    """
    sim: SimulationCfg = SimulationCfg()
    decimation: int = 1
    scene: SceneCfg = SceneCfg(env_spacing=2.5)
    use_domain_randomisation: bool = False
    episode_length_s: int = 10
    observation_space: int = 1
    action_space: int = 1
    mode: str = "train"

    def __post_init__(self):
        self.events = (
            RandomEventCfg() if self.use_domain_randomisation else ResetEnvCfg()
        )

class Env(DirectRLEnv):
    """
    Load the environment
    """
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg):
        self.mode = cfg.mode
        self.max_sim_steps = int(cfg.episode_length_s / cfg.sim.dt)
        self.sim_step_count = 0
        super().__init__(cfg)

    def _setup_scene(self):
        pass

    def _get_dones(self):
        """
        [REQUIRED] Detail the criteria by which a task is successfully completed.

        Returns:
        - task_complete, timeout (tuple(tensor, tensor))
        """
        task_complete = torch.tensor(False)
        timeout = self.sim_step_count > self.max_sim_steps
        return task_complete, timeout

    def _get_observations(self):
        return None

    @property
    def env_name(self):
        return "env_name" # e.g., room, kitchen
    
    @property
    def task_name(self):
        return "task_name" #e.g., stack_blocks, lift_blocks

    @property
    def task_language_instruction(self):
        return "instructions" # e.g., "Stack the red block on top of the red block"

    @property
    def task_phases(self):
        return [
            "list",
            "task", 
            "phases",
            "here"
        ] # e.g., ["reach for the block", "grasp the block", "lift the block"]
