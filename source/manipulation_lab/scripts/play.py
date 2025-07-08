import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManipulationLab")

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 2")
parser.add_argument("--task", type=str, default="Isaac-Blocks-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--teleop", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Remove unknown args to avoid Hydra conflicts
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Patch IsaacLab functions to avoid errors with Hydra
import manipulation_lab.scripts.utils._validation_patch
import manipulation_lab.scripts.utils._resolve_names_patch
logger.info("Patched string_utils.resolve_matching_names_values with manipulation_lab.scripts.utils._resolve_names_patch._patched_resolve_names_values")
logger.info("Patched configclass._validate with manipulation_lab.scripts.utils._validation_patch._patched_validate")

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Internal imports
from manipulation_lab.scripts.utils.runner import TaskRunner
from manipulation_lab.scripts.utils.teleop_handler import TeleopHandler

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cfg.num_envs = args_cli.num_envs
    cfg.device = args_cli.device

    # Create the environment from Hydra config
    env = instantiate(cfg.task)

    sim = env.unwrapped.sim
    scene = env.unwrapped.scene

    # Allow the simulation to warm up
    settle_steps = int(3.0 / sim.get_physics_dt())
    logger.info(f"Warming up the simulator ...")
    for _ in range(settle_steps):
        sim.step()
        sim_dt = sim.get_physics_dt()
        scene.update(sim_dt)
    logger.info(f"Simulator warm")

    if not cfg.teleop:
        runner = TaskRunner(cfg, env)
        runner.run(simulation_app=simulation_app)
    else:
        teleop_handler = TeleopHandler(env)
        teleop_handler.run_teleop(simulation_app=simulation_app)

if __name__ == "__main__":
    main()
    simulation_app.close()