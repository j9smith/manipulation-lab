# Config imports
from isaaclab.utils import configclass
from omegaconf import MISSING

# Env imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp.events as events
import isaaclab.sim as sim_utils

# Scene imports
from isaaclab.scene import InteractiveSceneCfg
from manipulation_lab.envs.packing_table.scene.packing_table import PackingTableSceneCfg

# Asset imports
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.assets import DeformableObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import manipulation_lab.envs.utils as utils
import torch

@configclass
class SceneCfg(InteractiveSceneCfg, PackingTableSceneCfg):
    """
    Design the scene by specifying prim configs to be constructed by the
    simulator.

    See for more details:
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html
    """
    robot: ArticulationCfg = MISSING

    wrist_camera: CameraCfg = MISSING

    scene_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/SceneCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 2.25),
            rot=(0.0, -0.45, 0.0, 0.89),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            horizontal_aperture=18,
            focus_distance=400.0,
            clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
        data_types=["rgb", "rgba", "depth"]
    )

    bear = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bear",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, 1.05)
        )
    )

    left_shoulder_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LeftShoulderCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(-0.1, 1.1, 1.9),
            rot=(0.88, 0.15, 0.20, -0.41),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256
    )

@configclass
class RandomEventCfg:
    """
    Define EventTermCfgs that are called by the simulator under certain conditions.
    Generally used for resets/randomisation. 

    See for more details:
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventTermCfg
    """
    # We always want to reset robot pose and velocity between episodes
    reset_robot_pose = EventTermCfg(
        func=utils.reset_robot_rand,
        mode="reset",
        params={
            "root_range": (0.05, 0.05),
            "joint_pos_range": (-0.1, 0.1)
        }
    )

    randomise_bear_placement = EventTermCfg(
        func=events.reset_nodal_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bear"),
            "position_range": {
                "x": (-0.05, 0.05),
                "y": (0.0, 0.10),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

@configclass
class ResetEnvCfg:
    """
    Deterministically reset the environment. Implemented as an 
    EventTermCfg (see above).
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
        self.mode = cfg.mode # str: generally "train"/"test" - but not constrained
        self.max_sim_steps = int(cfg.episode_length_s / cfg.sim.dt)
        self.sim_step_count = 0
        super().__init__(cfg)

    def _setup_scene(self):
        """
        Called on environment instantation. Include any logic here that programatically
        alters the scene outside of config. For example: change the spawn positions of
        sensors, articulations, or objects; define train/test splits using self.mode;
        programatically add new elements to the scene; etc.
        """
        robot = self.scene.articulations["robot"]
        robot.cfg.init_state.pos = (0.0, 0.35, 1.0)

    def get_dones(self):
        """
        [REQUIRED] Detail the criteria by which a task is successfully completed.

        Returns:
        - task_complete, timeout (tuple(tensor, tensor)) 
        """
        bear_xyz = self.scene.deformable_objects["bear"].data.root_pos_w[0]

        bounds_min = torch.tensor([-0.2, -0.9, 1.0], device=bear_xyz.device)
        bounds_max = torch.tensor([0.08, -0.3, 1.2], device=bear_xyz.device)

        task_complete = ((bear_xyz >= bounds_min) & (bear_xyz <= bounds_max)).all()

        if task_complete: print("Task complete!")

        timeout = self.sim_step_count > self.max_sim_steps
        return task_complete, timeout

    def _get_observations(self):
        """
        Not required and never handled - included to circumvent Isaac Lab
        NotImplementedError on environment reset.
        """
        return None

    @property
    def env_name(self) -> str:
        """
        Type: str
        The name of the environment in which the task exists. Used for
        dataset labelling.

        e.g., "room", "kitchen"
        """
        return "packing_table" 
    
    @property
    def task_name(self) -> str:
        """
        Type: str
        The name of the task used for dataset labelling. 

        e.g., "stack_blocks", "lift_blocks"
        """
        return "place_bear" 
    
    @property
    def task_language_instruction(self) -> str:
        """
        Type: str
        A single instruction detailing the task. Used for episode
        labelling and training language-aware models.

        e.g. "Stack the red block on top of the red block"
        """
        return "place the stuffed bear in the grey crate"

    @property
    def task_phases(self) -> list[str]:
        """
        Type: list[str]
        A list of the different phases denoted by language instructions
        inside the task. Used for episode labelling and training language-aware
        models.

        e.g. ["reach for the block", "grasp the block", "lift the block"]
        """
        return [
            "reach for the bear",
            "pick up the bear",
            "place the bear in the crate"
        ]
