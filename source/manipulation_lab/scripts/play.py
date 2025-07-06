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

import hydra
from omegaconf import DictConfig
from isaaclab_tasks.utils import parse_env_cfg

from manipulation_lab.scripts.handlers.runner import TaskRunner
from manipulation_lab.scripts.handlers.teleop_handler import TeleopHandler

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_cfg = parse_env_cfg(
    args_cli.task,
    device = args_cli.device,
    num_envs=args_cli.num_envs
    )

    if not args_cli.teleop:
        runner = TaskRunner(cfg)
        runner.run(simulation_app=simulation_app, 
                env_cfg=env_cfg,
                task=args_cli.task)
    else:
        teleop_handler = TeleopHandler()
        teleop_handler.run_teleop(simulation_app=simulation_app,
        env_cfg=env_cfg,
        task=args_cli.task)

if __name__ == "__main__":
    main()
    simulation_app.close()