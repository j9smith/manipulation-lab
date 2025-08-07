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
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.camera import CameraCfg

import torch
import manipulation_lab.envs.utils as utils

@configclass
class PushBlocksSceneCfg(InteractiveSceneCfg, RoomSceneCfg):
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

    scene_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/SceneCamera",
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.0, 1.25),
                                   rot=(0.0, -0.45, 0.0, 0.89), 
                                   convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, 
            focus_distance=400.0, 
            horizontal_aperture=18,#20.955, 
            clipping_range=(0.1, 1.0e5)),
        width=256,
        height=256
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.05)),
    )

    cone_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftCone",
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, -0.30, 0.05)),
    )

    cone_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightCone",
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, -0.30, 0.05)),
    )

@configclass
class BlocksEventCfg:
    randomise_blue_cube_placement = EventTermCfg(
        func=events.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cuboid_blue"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.1, 0.0),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

    randomise_right_cone_placement = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cone_right"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (0.0, 0.10),
                "z": (0.0, 0.0)
            },
            "velocity_range": {}
        }
    )

    randomise_cone_left_placement = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cone_left"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (0.0, 0.10),
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
class PushBlocksEnvCfg(DirectRLEnvCfg):
    """
    Create a config for the environment
    """
    sim: SimulationCfg = SimulationCfg()
    decimation: int = 1
    scene: PushBlocksSceneCfg = PushBlocksSceneCfg(env_spacing=2.5)
    use_domain_randomisation: bool = False
    episode_length_s: int = 10
    observation_space: int = 1
    action_space: int = 1
    mode: str = "train"

    def __post_init__(self):
        self.events = (
            BlocksEventCfg() if self.use_domain_randomisation else ResetEnvCfg()
        )

class PushBlocksEnv(DirectRLEnv):
    """
    Load the environment
    """
    cfg: PushBlocksEnvCfg

    def __init__(self, cfg: PushBlocksEnvCfg, mode: str = "train"):
        self.mode = cfg.mode
        self.max_sim_steps = int(cfg.episode_length_s / cfg.sim.dt)
        self.sim_step_count = 0
        super().__init__(cfg)

    def _setup_scene(self):
        if self.mode == "train":
            self.scene.rigid_objects["cuboid_blue"].cfg.init_state.pos = (0.3, 0.1, 0.05)
            self.scene.rigid_objects["cone_right"].cfg.init_state.pos = (0.1, -0.3, 0.0)
            self.scene.rigid_objects["cone_left"].cfg.init_state.pos = (0.4, -0.3, 0.0)
        else:
            self.scene.rigid_objects["cuboid_blue"].cfg.init_state.pos = (-0.3, 0.0, 0.05)
            self.scene.rigid_objects["cone_right"].cfg.init_state.pos = (-0.4, -0.3, 0.0)
            self.scene.rigid_objects["cone_left"].cfg.init_state.pos = (-0.1, -0.3, 0.0)

    def _get_observations(self):
        pass
    
    def get_dones(self):
        l_cone_pos = self.scene.rigid_objects["cone_left"].data.root_link_pose_w[0, :2]
        r_cone_pos = self.scene.rigid_objects["cone_right"].data.root_link_pose_w[0, :2]

        cube_pos = self.scene.rigid_objects["cuboid_blue"].data.root_link_pose_w[0, :2]

        finish_line = l_cone_pos - r_cone_pos
        normal = torch.tensor([-finish_line[1], finish_line[0]], device=cube_pos.device)

        cube_vec = cube_pos - l_cone_pos
        distance = torch.dot(cube_vec, normal)

        task_complete = distance <= 0 and cube_pos[0] > r_cone_pos[0] and cube_pos[0] < l_cone_pos[0]
        timeout = self.sim_step_count > self.max_sim_steps
        return task_complete, timeout

    @property
    def env_name(self):
        return "room"
    
    @property
    def task_name(self):
        return "push_blocks"

    @property
    def task_language_instruction(self):
        return "push the blue block between the two red cones"

    @property
    def task_phases(self):
        return [
            "reach for the blue block",
            "push the blue block between the red cones"
        ]
