import logging
logger = logging.getLogger("ManipulationLab.ModelHandler")

import torch
import numpy as np
from hydra.utils import instantiate

class ModelHandler:
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        device = self.cfg.controller.device
        logger.info("Loading controller ...")

        # Load policy model with defined weights
        self.model = instantiate(self.cfg.controller.model)
        if self.cfg.controller.model_weights is not None:
                logger.info(f"Loading model weights from {self.cfg.controller.model_weights}")
                self.model.load_state_dict(
                    torch.load(self.cfg.controller.model_weights, map_location=device, weights_only=True)
                )
        else:
            logger.warning(
                "No model weights specified. Ensure this was intentional."
            )
        self.model.to(device).eval()

        # Load image encoder if defined
        if self.cfg.controller.encoder is not None:
            self.encoder = instantiate(self.cfg.controller.encoder)

            self.encoder.to(device).eval()
        else:
            self.encoder = None

        self.cfg = cfg
        self.camera_keys = cfg.controller.camera_keys
        self.proprio_keys = cfg.controller.proprio_keys
        self.sensor_keys = cfg.controller.sensor_keys
        self.model_use_structured_obs = self.cfg.controller.model_use_structured_obs

    def _extract_desired_obs(self, obs: dict):
     """
     Extracts the desired observation from the raw observation dictionary.
     """
     # TODO: What do we do if images are different dims? What if we want sensor and proprio data too?
     def _get_nested_data(nested_dict: dict, key: str):
         keys = key.split("/")
         for k in keys:
             nested_dict = nested_dict[k]
         return nested_dict

     camera_obs = {}

     if self.camera_keys:
         for key in self.camera_keys:
             camera_obs[key] = _get_nested_data(obs, key)

     proprio_obs = {}
     if self.proprio_keys:
         for key in self.proprio_keys:
             proprio_obs[key] = _get_nested_data(obs, key)

     sensor_obs = {}
     if self.sensor_keys:
         for key in self.sensor_keys:
             sensor_obs[key] = _get_nested_data(obs, key)

     return camera_obs, proprio_obs, sensor_obs

    def forward(self, raw_obs):
        camera_obs, proprio_obs, sensor_obs = self._extract_desired_obs(raw_obs)

        obs = []

        if self.model_use_structured_obs == False:
            # Process all target data and append to obs list
            if self.encoder is not None:
                for _, value in camera_obs.items():
                    value = torch.tensor(value, dtype=torch.float32).permute(2, 0, 1) / 255.0 # Normalize to [0, 1]
                    value = value.unsqueeze(0) # C, H, W -> B, C, H, W
                    value = value.to(self.cfg.controller.device)
                    cam_latent_obs = self.encoder(value)
                    if cam_latent_obs.ndim == 2: cam_latent_obs = cam_latent_obs.squeeze(0)
                    obs.append(cam_latent_obs)

            if self.proprio_keys:
                for _, value in proprio_obs.items():
                    value = torch.tensor(value).to(self.cfg.controller.device)
                    obs.append(value)   

            if self.sensor_keys:
                # TODO: How do we handle multidimensional sensor data?
                for _, value in sensor_obs.items():
                    value = torch.tensor(value).to(self.cfg.controller.device)
                    obs.append(value)

            # Concatenate all target data into flat tensor
            obs = torch.cat(obs, dim=-1)

        else: obs = (camera_obs, proprio_obs, sensor_obs)

        if self.model_use_structured_obs == False and obs.ndim == 1:
            obs = obs.unsqueeze(0)

        # Run the model
        with torch.no_grad():
            actions = self.model(obs)

        return actions