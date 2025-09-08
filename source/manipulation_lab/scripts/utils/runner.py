import logging
logger = logging.getLogger("ManipulationLab.TaskRunner")

from hydra.utils import instantiate
from manipulation_lab.scripts.control.action_handler import ActionHandler
from manipulation_lab.scripts.control.obs_handler import ObservationHandler
from manipulation_lab.scripts.control.controller import Controller
import time
import wandb
from omegaconf import OmegaConf
from statistics import mean
import sys

from threading import Event

class TaskRunner:
    """
    Build and execute task logic
    """
    def __init__(self, cfg, env):
        # Environment variables
        self.cfg = cfg
        self.env = env
        self.env.seed(self.cfg.environment_seed)
        self.scene = self.env.unwrapped.scene
        self.sim = self.env.unwrapped.sim
        self.sim_dt = self.sim.get_physics_dt()
        self.control_event = Event()

        # Control handlers
        use_oracle_obs = True if self.cfg.controller.oracle_keys else False
        self.obs_handler = ObservationHandler(env=self.env, use_oracle=use_oracle_obs)
        self.action_handler = ActionHandler(env=self.env, control_mode=self.cfg.controller.action_space)

        # Models
        self.model = None
        self.encoder = None
        self.controller = Controller(
            cfg=self.cfg,
            control_freq=self.cfg.controller.control_frequency,
            control_event=self.control_event,
            sim_dt=self.sim_dt,
        )

        self.step_count = 0

    def run(self, simulation_app):
        """
        Runs the simulation loop.
        """
        config = OmegaConf.to_container(
            self.cfg, 
            resolve=True, 
            throw_on_missing=True
        )

        suffix = str(input(
            f"Please enter a suffix for this run: {self.env.env_name}_{self.env.task_name}_"
            )
        )
        wandb_run_name = f"{self.env.env_name}_{self.env.task_name}_{suffix}"

        run = wandb.init(
        project="Manipulation Lab - Evaluation",
        name=wandb_run_name,
        config=config
        )

        results_table = wandb.Table(
            columns=["Attempt", "Result", "Time Taken (s)"]
        )

        self._reset_scene()
        self.controller.start()

        attempts = 1
        successes = 0
        success_times = []

        start_time = time.time()

        while simulation_app.is_running() and attempts <= self.cfg.max_attempts:
            # Step simulator and update sim time
            self.sim.step()
            self.step_count += 1
            self.controller.sim_step_count = self.step_count
            self.env.sim_step_count = self.step_count

            # Update buffers to reflect new sim state
            self.scene.update(self.sim_dt)

            # Get observations and push to controller
            obs = self.obs_handler.get_obs()
            self.controller.update_obs(obs)

            # Inform the control loop that the sim has stepped
            self.control_event.set()
            
            # Check for new actions from controller
            action = self.controller.get_action()
            if action is not None:
                self.action_handler.apply(action=action)

            task_complete, timeout = self.env.get_dones()

            if task_complete:
                successes += 1
                time_taken = self.step_count * self.sim_dt
                success_times.append(time_taken)

                results_table.add_data(attempts, "Success", round(time_taken, 2))

                logger.info(
                    f"Task completed successfully! Score: {successes} / {attempts}, "
                    f"time taken {time_taken:.2f}s"
                )
                attempts += 1
                time.sleep(2.5)
                self._reset_scene()

            if timeout:
                results_table.add_data(attempts, "Failure", None)

                logger.info(f"Timeout - task failed. Score: {successes} / {attempts}")
                attempts += 1
                time.sleep(2.5)
                self._reset_scene()

        success_rate = successes / self.cfg.max_attempts
        average_success_time = mean(success_times) if success_times else 0.0

        logger.info(
            f"Task successes: {successes} / {self.cfg.max_attempts}"
        )

        wandb.log({
            "evaluation/results_table": results_table,
        })

        wandb.run.summary["scene"] = self.env.env_name
        wandb.run.summary["task"] = self.env.task_name
        wandb.run.summary["weights"] = self.cfg.controller.model_weights
        wandb.run.summary["success_rate"] = success_rate
        wandb.run.summary["average_success_time"] = average_success_time

        wandb.finish()

        logger.info(
            f"Time taken: {time.time() - start_time:.2f}s"
        )
        simulation_app.close()
        sys.exit()

    def _reset_scene(self):
        # TODO: Do some ablations here to find out what we can get rid of
        self.env.reset()
        self.sim.step()

        self.scene.update(self.sim_dt)
        robot = self.scene.articulations["robot"]
        robot.set_joint_position_target(robot.data.joint_pos)
        self.scene.write_data_to_sim()
        
        self.step_count = 0
        self.controller.reset()
            
        logger.info("Scene reset.")