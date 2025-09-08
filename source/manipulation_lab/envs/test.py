# Config imports
from isaaclab.utils import configclass
from omegaconf import MISSING

# Env imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp.events as events

# Scene imports
from isaaclab.scene import InteractiveSceneCfg

# Asset imports
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCameraCfg, ImuCfg, RayCasterCamera, ContactSensor, Imu
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg, GridPatternCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import DeformableObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import manipulation_lab.envs.utils as utils
import isaaclab.sim as sim_utils
import torch

from manipulation_lab.envs.packing_table.scene.packing_table import PackingTableSceneCfg
from manipulation_lab.envs.room.scene.room_scene import RoomSceneCfg


@configclass
class SceneCfg(InteractiveSceneCfg, PackingTableSceneCfg):
    """
    Design the scene by specifying prim configs to be constructed by the
    simulator.

    See for more details:
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html
    """
    robot: ArticulationCfg = MISSING

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    scene_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/SceneCamera",
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.0, 1.25),
                                   rot=(0.0, -0.45, 0.0, 0.89), 
                                   convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, 
            focus_distance=400.0, 
            horizontal_aperture=18, 
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256,
        data_types=["rgb", "depth", "rgba", "semantic_segmentation", "instance_segmentation_fast"]
    )

    other_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/OtherCamera",
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.0, 1.25),
                                   rot=(0.0, -0.45, 0.0, 0.89), 
                                   convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, 
            focus_distance=400.0, 
            horizontal_aperture=18, 
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256,
        data_types=["rgb"]
    )

    # height_scanner = RayCasterCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/",
    #     update_period=0.02,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/defaultGroundPlane"],
    #     attach_yaw_only=True
    # )

    # cuboid_blue = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CuboidBlue",
    #     spawn=sim_utils.MeshCuboidCfg(
    #         size=(0.05, 0.05, 0.05),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg()
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.2, 0.05)),
    # )


    # bear = DeformableObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Bear",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
    #         deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
    #     ),
    #     init_state=DeformableObjectCfg.InitialStateCfg(
    #         pos=(0.0, 0.1, 0.05)
    #     )
    # )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.5, 0.5, 0.5)),
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
        func=utils.reset_robot,
        mode="reset",
        params={}
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

        robot.cfg.init_state.pos = (0.0, 0.0, -5.0)

        # ray_caster_cfg = RayCasterCameraCfg(
        #     prim_path="/World/envs/env_0/Sensors/RayCasterCamera",
        #     mesh_prim_paths=["/World/defaultGroundPlane"],
        #     offset=RayCasterCameraCfg.OffsetCfg(
        #         pos=(0.08, 0.0, 0.0),
        #         rot=(0.0, 0.0, 0.0, 1.0),
        #     ),
        #     pattern_cfg=PinholeCameraPatternCfg(width=64, height=48),
        # )
        # RayCasterCamera(ray_caster_cfg)

        # contact_sensor_cfg = ContactSensorCfg(
        #     prim_path="/World/envs/env_0//Robot/panda_hand"
        # )
        # ContactSensor(contact_sensor_cfg)

        # imu_cfg = ImuCfg(
        #     prim_path="/World/envs/env_0//Robot/panda_hand/imu",
        # )
        # Imu(imu_cfg)

    def get_dones(self):
        """
        [REQUIRED] Detail the criteria by which a task is successfully completed.

        Returns:
        - task_complete, timeout (tuple(tensor, tensor)) 
        """
        task_complete = torch.tensor(False)
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
        return "env_name" 
    
    @property
    def task_name(self) -> str:
        """
        Type: str
        The name of the task used for dataset labelling. 

        e.g., "stack_blocks", "lift_blocks"
        """
        return "task_name" 
    
    @property
    def task_language_instruction(self) -> str:
        """
        Type: str
        A single instruction detailing the task. Used for episode
        labelling and training language-aware models.

        e.g. "Stack the red block on top of the red block"
        """
        return "instructions"

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
            "list",
            "task", 
            "phases",
            "here"
        ]
