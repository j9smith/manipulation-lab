from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG as FRANKA_PANDA_CFG
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.sensors.camera import CameraCfg

from manipulation_lab.envs.tabletop.scene.tabletop_cfg import TableTopSceneCfg

import math

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

from omegaconf import MISSING

@configclass
class BlocksSceneCfg(InteractiveSceneCfg, TableTopSceneCfg):
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
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.6, 0.6),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 1.06)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 1.06)),
    )

@configclass
class BlocksEnvCfg(DirectRLEnvCfg):
    """
    Create a config for the environment
    """
    sim: SimulationCfg = SimulationCfg()
    decimation: int = 1
    scene: BlocksSceneCfg = BlocksSceneCfg(env_spacing=2.5)
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

    def _reset_idx(self, env_ids):
        robot = self.scene.articulations["robot"]

        default_joint_pos = robot.data.default_joint_pos[env_ids]
        default_joint_vel = robot.data.default_joint_vel[env_ids]

        robot.write_joint_state_to_sim(
            position=default_joint_pos,
            velocity=default_joint_vel,
            env_ids=env_ids
        )

        for object_name in self.scene.rigid_objects.keys():
            obj = self.scene.rigid_objects[object_name]
            default_pos = obj.data.default_root_state[env_ids]
            obj.write_root_state_to_sim(default_pos, env_ids=env_ids)

    def _setup_scene(self):
        pass

    def _get_observations(self):
        return None


    @property
    def env_name(self):
        return "tabletop"
    
    @property
    def task_name(self):
        return "blocks"
