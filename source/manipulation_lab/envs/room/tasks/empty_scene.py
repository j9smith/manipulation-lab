import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empty stage")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app
simulation_app.update()

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import torch
from isaaclab.assets import Articulation

from manipulation_lab.assets.robots.franka import FRANKA_PANDA_CFG
from isaacsim.core.prims import SingleArticulation
import isaacsim.core.utils.prims as prim_utils

def design_scene():
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=7000.0,
        color=(0.75, 0.75, 0.75)
    )

    #cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    cfg_light_distant.func("/World/Lighting/Sun", cfg_light_distant, translation=(1, 0, 10.0))

    spotlight = sim_utils.SphereLightCfg(
        intensity=5000.0,
        radius=0.1,
        color=(1.0, 1.0, 1.0)
    )
    spotlight.func("/World/Lighting/Overhead", spotlight, translation=(0.0, 0.0, 5))

    cfg_cuboid_red= sim_utils.MeshCuboidCfg(
        size=(0.05, 0.05, 0.05),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg()
    )
    cfg_cuboid_red.func("/World/Objects/CuboidRed", cfg_cuboid_red, translation=(0.15, 0.0, 1.06))

    cfg_cuboid_blue= sim_utils.MeshCuboidCfg(
        size=(0.05, 0.05, 0.05),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg()
    )
    cfg_cuboid_blue.func("/World/Objects/CuboidBlue", cfg_cuboid_blue, translation=(-0.15, 0.0, 1.06))

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

    # franka_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FactoryFranka/factory_franka_instanceable.usd")                


    prim_utils.create_prim("/World/Origin1", "Xform", translation=[-0.55, 0.0, 1.06])
    franka_cfg = FRANKA_PANDA_CFG.copy()
    franka_cfg.prim_path = "/World/Origin1/Franka"
    franka = Articulation(cfg=franka_cfg)

    return franka

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

def stack_sequence(sim, franka):
    device = franka.device if hasattr(franka, "device") else "cuda:0"

    def step_n(n=1):
        for _ in range(n):
            sim.step()
            time.sleep(0.01)
    
    home = franka.data.default_joint_pos.clone().to(device) #torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8])
    reach_block = torch.tensor([0.0, -0.8, 0.0, -1.8, 0.0, 1.8, 0.6])
    lift_block = torch.tensor([0.0, -0.4, 0.0, -2.4, 0.0, 2.4, 1.0])
    over_target = torch.tensor([0.4, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8])

    open_grip = torch.tensor([0.04, 0.04])
    closed_grip = torch.tensor([0.0, 0.0])

    def apply(joint_pos):
        #action = ArticulationAction(joint_positions=joint_pos)
        #franka.apply_action(action)
        print(f"Attempting {joint_pos}")
        franka.set_joint_position_target(joint_pos)#, franka.data.default_joint_vel.clone())
        franka.write_data_to_sim()
        sim.step()
        franka.update(sim.get_physics_dt())
        step_n(100)

    apply(torch.cat([reach_block, open_grip]))
    apply(torch.cat([reach_block, closed_grip]))
    apply(torch.cat([lift_block, closed_grip]))
    apply(torch.cat([over_target, closed_grip]))
    apply(torch.cat([torch.tensor([0.4, -0.8, 0.0, -1.8, 0.0, 1.8, 0.6]), closed_grip]))
    apply(torch.cat([torch.tensor([0.4, -0.8, 0.0, -1.8, 0.0, 1.8, 0.6]), open_grip]))

    apply(torch.cat([home, open_grip]))

from isaaclab.sim import SimulationCfg, SimulationContext

import time

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 5], [0.0, 0.0, 1.06])

    franka = design_scene()

    sim.reset()
    stack_sequence(sim, franka)

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()

    simulation_app.close()