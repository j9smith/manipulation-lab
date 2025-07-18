from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp.events as events

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG as FRANKA_PANDA_CFG
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.sensors.camera import CameraCfg

from manipulation_lab.envs.room.scene.room_scene import RoomSceneCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

from omegaconf import MISSING

@configclass
class BlocksSceneCfg(InteractiveSceneCfg, RoomSceneCfg):
    """
    Design the scene
    """
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.5, 0.5, 0.5))
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
            horizontal_aperture=10, #20.955, 
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

    cuboid_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CuboidRed",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.05)),
    )

    cuboid_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CuboidBlue",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.05)),
    )

def test(env, env_ids):
    print(f"env ids: {env_ids}")

@configclass
class BlocksEventCfg:
    randomise_red_cube_placement = EventTermCfg(
        func=events.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cuboid_red"),# SceneEntityCfg("cuboid_blue")],
            "pose_range": {
                "x": (-0.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

    randomise_blue_cube_placement = EventTermCfg(
        func=events.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cuboid_blue"),
            "pose_range": {
                "x": (-0.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

    randomise_robot_pose = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.025, 0.025),
            "velocity_range": (0.0, 0.0)
        }
    )

    test = EventTermCfg(
        func=test,
        mode="reset",
        params={}
    )

@configclass
class BlocksEnvCfg(DirectRLEnvCfg):
    """
    Create a config for the environment
    """
    sim: SimulationCfg = SimulationCfg()
    decimation: int = 1
    scene: BlocksSceneCfg = BlocksSceneCfg(env_spacing=2.5)
    events = BlocksEventCfg()
    episode_length_s: int = 60
    observation_space: int = 1
    action_space: int = 1

class BlocksEnv(DirectRLEnv):
    """
    Load the environment
    """
    cfg: BlocksEnvCfg

    def __init__(self, cfg: BlocksEnvCfg):
        super().__init__(cfg)

    def _setup_scene(self):
        pass

    def _get_observations(self):
        return None


    @property
    def env_name(self):
        return "room"
    
    @property
    def task_name(self):
        return "blocks"
