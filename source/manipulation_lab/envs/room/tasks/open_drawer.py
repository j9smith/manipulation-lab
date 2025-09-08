# Config imports
from isaaclab.utils import configclass
from omegaconf import MISSING

# Env imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp.events as events

# Scene imports
from isaaclab.scene import InteractiveSceneCfg
from manipulation_lab.envs.room.scene.room_scene import RoomSceneCfg

# Asset imports
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.assets import RigidObjectCfg

import torch
import manipulation_lab.envs.utils as utils

@configclass
class SceneCfg(InteractiveSceneCfg, RoomSceneCfg):
    """
    Design the scene
    """
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.5, 0.5, 0.5)),
    )

    robot: ArticulationCfg = MISSING

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, -0.06), #x = vertical, y = horizontal, z = forwards(+)/backwards(-)
                                   rot=(0.0, 0.60, 0.0, 0.80),
                                   convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10, 
            focus_distance=400.0, 
            horizontal_aperture=10,
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256
    )

    left_shoulder_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/LeftShoulderCamera",
        offset=CameraCfg.OffsetCfg(pos=(-0.1, 1.1, 0.9),
                                   rot=(0.88, 0.15, 0.20, -0.41), 
                                   convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256
    )
    
    right_shoulder_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/RightShoulderCamera",
        offset=CameraCfg.OffsetCfg(pos=(-0.1, -0.8, 0.5),
                                   rot=(0.87, -0.14, 0.21, 0.43), 
                                   convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256
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
        height=256
    )

    drawer: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 2.0, -0.4),
            rot=(0.71, 0.0, 0.0, 0.71),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                    effort_limit_sim=150.0,
                    stiffness=10.0,
                    damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=150.0,
                stiffness=0.0,
                damping=0.0,
            )
        }
    )

@configclass
class RandomEventCfg:
    reset_robot_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

    reset_robot_pose = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0)
        }
    )

@configclass
class ResetEnvCfg:
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

    def _get_observations(self):
        pass

    def _setup_scene(self):
        robot = self.scene.articulations["robot"]
        print(robot.cfg.init_state.pos)
        if self.mode == "train":
            robot.cfg.init_state.pos = (0.0, 3.25, -0.8)
        else:
            robot.cfg.init_state.pos = (0.0, 3.25, -0.8)
        print(robot.cfg.init_state.pos)
    
    def get_dones(self):
        task_complete, timeout = (torch.Tensor(0), torch.Tensor(0))
        return task_complete, timeout

    @property
    def env_name(self):
        return "room"
    
    @property
    def task_name(self):
        return "open_drawer"

    @property
    def task_language_instruction(self):
        return "open the drawer"

    @property
    def task_phases(self):
        return [
            "reach for the drawer handle",
            "pull the drawer open"
        ]
