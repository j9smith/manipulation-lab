import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 2")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg

import torch

from manipulation_lab.assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import subtract_frame_transforms

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05))
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.5, 0.5, 0.5))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    )

    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(cfg=diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
            ]
    
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    current_goal_idx = 0

    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    franka_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"], body_names=["panda_hand"])

    franka_cfg.resolve(scene)

    ee_jacobi_idx = franka_cfg.body_ids[0] - 1

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 150 == 0: 
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()

            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            ik_commands[:] = ee_goals[current_goal_idx]

            joint_pos_des = joint_pos[:, franka_cfg.joint_ids].clone()

            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, franka_cfg.joint_ids]
            ee_pose_w = robot.data.body_link_pose_w[:, franka_cfg.body_ids[0]] 
            root_pose_w = robot.data.root_link_pose_w 
            joint_pos = robot.data.joint_pos[:, franka_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=franka_cfg.joint_ids)
        scene.write_data_to_sim()

        sim.step()

        count += 1
        
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, franka_cfg.body_ids[0], 0:7]

        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()