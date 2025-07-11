import logging
logger = logging.getLogger("ManipulationLab")

# Launch the simulation app
from isaaclab.app import AppLauncher
launch_cfg = {
    "headless": False,
    "enable_cameras": True,
    "num_envs": 1,
    "device": "cuda",
}

app_launcher = AppLauncher(launcher_args=launch_cfg)
simulation_app = app_launcher.app

# Patch IsaacLab functions to avoid errors with Hydra
# Must execute after app launch
import manipulation_lab.scripts.patch._validation_patch
import manipulation_lab.scripts.patch._resolve_names_patch
logger.info("Patched string_utils.resolve_matching_names_values with manipulation_lab.scripts.utils._resolve_names_patch._patched_resolve_names_values")
logger.info("Patched configclass._validate with manipulation_lab.scripts.utils._validation_patch._patched_validate")

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Internal imports
from manipulation_lab.scripts.utils.runner import TaskRunner
from manipulation_lab.scripts.teleop.teleop_handler import TeleopHandler

@hydra.main(config_path="../config/play", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # Create the environment from Hydra config
    env = instantiate(cfg.task)

    sim = env.unwrapped.sim
    scene = env.unwrapped.scene

    # Allow the simulation to warm up
    settle_steps = int(1.5 / sim.get_physics_dt())
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
        teleop_handler = TeleopHandler(env, cfg)
        teleop_handler.run_teleop(simulation_app=simulation_app)

if __name__ == "__main__":
    main()
    simulation_app.close()