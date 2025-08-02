import logging
logger = logging.getLogger("ManipulationLab")


from isaaclab.app import AppLauncher
# Launch the simulation app
launch_cfg = {
    "headless": False,
    "enable_cameras": True,
    "num_envs": 1,
    "device": "cuda",
}
app_launcher = AppLauncher(launcher_args=launch_cfg)
simulation_app = app_launcher.app

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Internal imports
from manipulation_lab.scripts.utils.runner import TaskRunner
from manipulation_lab.scripts.teleop.teleop_handler import TeleopHandler
from manipulation_lab.scripts.utils.open_loop import OpenLoopHandler

# Patch IsaacLab functions to avoid errors with Hydra
# Must execute after app launch
import manipulation_lab.scripts.patch._validation_patch
import manipulation_lab.scripts.patch._resolve_names_patch
logger.info("Patched string_utils.resolve_matching_names_values with manipulation_lab.scripts.utils._resolve_names_patch._patched_resolve_names_values")
logger.info("Patched configclass._validate with manipulation_lab.scripts.utils._validation_patch._patched_validate")


@hydra.main(config_path="../config/", config_name="play_config", version_base=None)
def main(cfg: DictConfig):
    # Create the environment from Hydra config
    if cfg.teleop or cfg.force_train_setup:
         cfg.task.cfg.mode = "train"
    else:
        cfg.task.cfg.mode = "test"

    env = instantiate(cfg.task)

    if not cfg.teleop:
            runner = TaskRunner(cfg, env)
            runner.run(simulation_app=simulation_app)
    else:
            teleop_handler = TeleopHandler(env, cfg)
            teleop_handler.run_teleop(simulation_app=simulation_app)
            simulation_app.close()

def replay_trajectory(env, sim, scene, sim_dt):
    import manipulation_lab.scripts.dataset.reader as reader
    reader = reader.DatasetReader(dataset_dirs=["/home/ubuntu/Projects/manipulation_lab/static_datasets/room/blocks/clean"])
    episode = reader.load_episode(0)
    import manipulation_lab.scripts.control.action_handler as action_handler
    action_handler = action_handler.ActionHandler(env, control_mode="delta_cartesian")
    import numpy as np
    import torch
    
    ee_deltas = episode["actions"]["expert"]["ee_deltas"]
    gripper_deltas = episode["actions"]["expert"]["gripper_deltas"]
    logger.info(f"ee deltas: {ee_deltas}")
    logger.info(f"gripper deltas: {gripper_deltas}")

    logger.info(f"Gripper deltas length: {len(gripper_deltas)}")
    logger.info(f"EE deltas length: {len(ee_deltas)}")

    sim_steps = 0

    for i in range(len(gripper_deltas)):
        for _ in range(2):
            action = np.concatenate([ee_deltas[i], [gripper_deltas[i]]])
            action = torch.tensor(action)
            action_handler.apply(action=action)
            sim.step()
            scene.update(sim_dt)
            sim_steps += 1

    logger.info(f"Simulation steps: {sim_steps}")

if __name__ == "__main__":
    
    main()