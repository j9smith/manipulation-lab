import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 2")
parser.add_argument("--task", type=str, default="Isaac-Blocks-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import hydra
from omegaconf import DictConfig
from isaaclab_tasks.utils import parse_env_cfg

from manipulation_lab.scripts.handlers.runner import TaskRunner

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_cfg = parse_env_cfg(
    args_cli.task,
    device = args_cli.device,
    num_envs=args_cli.num_envs
    )

    runner = TaskRunner(cfg)
    runner.run(simulation_app=simulation_app, 
               env_cfg=env_cfg,
               task=args_cli.task)

if __name__ == "__main__":
    main()
    simulation_app.close()