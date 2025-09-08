import logging
logger = logging.getLogger("ManipulationLab.OpenLoop")

import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

import manipulation_lab.scripts.dataset.reader as reader
import manipulation_lab.scripts.control.model_handler as model_handler

class OpenLoopHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reader = reader.DatasetReader(dataset_dirs=self.cfg.openloop.dataset_dir)
        self.episode = self.reader.load_episode(self.cfg.openloop.episode_idx)

        self.model_handler = model_handler.ModelHandler(cfg=cfg)

    def run_open_loop(self):
        obs = self.episode["observations"]
        num_frames = len(self.episode["sim_time"])

        ee_actions = self.episode["actions"]["expert"]["ee_deltas"]
        gripper_actions = self.episode["actions"]["expert"]["gripper_deltas"]

        actual_actions = np.hstack([ee_actions, gripper_actions[:, None]])
        predicted_actions = []

        camera_keys = self.cfg.controller.camera_keys if self.cfg.controller.camera_keys is not None else []
        proprio_keys = self.cfg.controller.proprio_keys if self.cfg.controller.proprio_keys is not None else []
        sensor_keys = self.cfg.controller.sensor_keys if self.cfg.controller.sensor_keys is not None else []

        keys = camera_keys + proprio_keys + sensor_keys

        def _get_nested_data(nested_dict: dict, key: str):
            keys = key.split("/")
            for k in keys:
                nested_dict = nested_dict[k]
            return nested_dict

        for i in range(num_frames):
            raw_obs = {}

            for key in keys:
                data = _get_nested_data(obs, key)
                frame = data[i]

                key_parts = key.split("/")
                current_key = raw_obs
                for part in key_parts[:-1]:
                    current_key = current_key.setdefault(part, {})
                current_key[key_parts[-1]] = frame
            pred = self.model_handler.forward(raw_obs).cpu().numpy()

            # TODO: We only take the first action of the chunk
            # Re-visit this and implement a more robust way to handle chunks 
            first_action = np.atleast_2d(pred)[0]
            predicted_actions.append(first_action)

            if i % 10 == 0:
                logger.info(f"Processing frame {i}/{num_frames}")

        actual = np.asarray(actual_actions)
        predicted = np.asarray(predicted_actions)

        assert actual.shape == predicted.shape, f"Shape mismatch: {actual.shape} != {predicted.shape}"

        mse = np.mean((actual - predicted) ** 2, axis=0)

        time, action_dim = actual.shape

        print("MSE:")
        for dimension in range(action_dim):
            print(f"Dim {dimension}: {mse[dimension]:.4f}")

        self.generate_plots(actual, predicted, time, action_dim)

        logger.info(f"Plots available in outputs/plots/")
    
    def generate_plots(self, actual, predicted, time_dim, action_dim):
        import os
        os.makedirs("outputs/plots", exist_ok=True)

        for dimension in range(action_dim):
            logger.info(f"Generating plot for dimension {dimension}")
            plt.figure(figsize=(6,4))
            plt.plot(actual[:, dimension],    label="Actual")
            plt.plot(predicted[:, dimension], label="Predicted")
            plt.title(f"Action dimension {dimension}")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/plots/action_dim_{dimension}.png")
            plt.close()

@hydra.main(config_path="../../config/", config_name="play_config", version_base=None)
def main(cfg: DictConfig):
    olp = OpenLoopHandler(cfg)
    olp.run_open_loop()

if __name__ == "__main__":
    main()