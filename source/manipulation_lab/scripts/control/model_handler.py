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
        self.device = self.cfg.device
        logger.info("Loading controller ...")

        # Load policy model with defined weights
        self.model = instantiate(self.cfg.model)
        if self.cfg.controller.model_weights is not None:
                logger.info(f"Loading model weights from {self.cfg.controller.model_weights}")
                try:
                    self.model.load_state_dict(
                        torch.load(self.cfg.controller.model_weights, map_location=self.device, weights_only=True)
                    )
                except Exception as e:
                    logger.warning(
                        f"Exception:\n{e}\nModel weights specified but model does not accept weights. "
                        f"Continuing without loading weights. Ensure this behaviour is intentional."
                    )
        else:
            logger.warning(
                "No model weights specified. Ensure this was intentional."
            )
        self.model.to(self.device).eval()
 
        # Load image encoder if defined
        if self.cfg.get("encoder", None):
            self.encoder = instantiate(self.cfg.encoder)

            self.encoder.to(self.device).eval()
        else:
            self.encoder = None

        self.cfg = cfg
        self.camera_keys = cfg.controller.camera_keys
        self.proprio_keys = cfg.controller.proprio_keys
        self.sensor_keys = cfg.controller.sensor_keys
        self.oracle_keys = cfg.controller.oracle_keys
        self.model_use_structured_obs = self.cfg.controller.model_use_structured_obs

    def _extract_desired_obs(self, obs: dict):
        """
        Extracts the desired observation from the raw observation dictionary.

        Parameters:
        - obs: dict - A dictionary of raw observations passed from ObsHandler of the format e.g.,
        {
            "sensors": {
                "wrist_camera": {
                        "rgb": nparray,
                        ...
                    },
            },
            "robot": {
                "joint_pos": nparray,
                ...
            }
        }

        Returns:
        - camera_obs: A flattened dictionary of { key: nparray }, e.g.
        {
            "sensors/camera_key/rgb": nparray
        }
        - proprio_obs: A flattened dictionary of { key: nparray }
        - sensor_obs: A flattened dictionary of { key: nparray }
        """

        # TODO: What do we do if images are different dims? What if we want sensor and proprio data too?
        def _get_nested_data(nested_dict: dict, key: str):
            """
            Extracts observation data from nested dictionaries.

            Parameters:
            - nested_dict: A dictionary of dictionaries.
            - key: The key referring to the data to resolve, e.g.
            "sensors/sensor_name/type" resolves nested_dict[sensors][sensor_name][type]

            Returns: The data contained at the leaf of the dictionary
            """
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

        oracle_obs = {}
        if self.oracle_keys:
            for key in self.oracle_keys:
                oracle_obs[key] = _get_nested_data(obs, key)

        return camera_obs, proprio_obs, sensor_obs, oracle_obs

    def forward(self, raw_obs: dict) -> torch.Tensor:
        """
        Performs a forward pass of the loaded model. If model takes structured obs, the forward pass
        will send a dictionary of {key: data,} to the model for handling. Otherwise, the forward pass
        will encode images, concatenate with other obs, and then send a tensor of shape (B, D_total)
        to the model. 

        Parameters:
        - raw_obs: dict - A dictionary of raw observations passed from ObsHandler of the format e.g.,
        {
            "sensors": {
                "wrist_camera": {
                        "rgb": nparray,
                        ...
                    },
            },
            "robot": {
                "joint_pos": nparray,
                ...
            }
        }

        Returns:
        - actions: torch.Tensor: A tensor of action commands of which the shape and type
        are defined by the model and specified by the user in config.
        """
        camera_obs, proprio_obs, sensor_obs, oracle_obs = self._extract_desired_obs(raw_obs)

        obs = []

        if self.model_use_structured_obs == False:
            # Process all target data and append to obs list
            if self.encoder is not None:
                for _, value in camera_obs.items():
                    value = torch.tensor(value, dtype=torch.float32).permute(2, 0, 1)
                    value = value.unsqueeze(0) # C, H, W -> B, C, H, W
                    value = value.to(self.device)
                    cam_latent_obs = self.encoder(value)

                    # If encoder returns (B, D), squeeze to (D) for concatenating
                    if cam_latent_obs.ndim == 2: cam_latent_obs = cam_latent_obs.squeeze(0)
                    obs.append(cam_latent_obs)

            if self.proprio_keys:
                for _, value in proprio_obs.items():
                    value = torch.tensor(value).to(self.device)
                    obs.append(value)

            if self.sensor_keys:
                # TODO: How do we handle multidimensional sensor data?
                for _, value in sensor_obs.items():
                    value = torch.tensor(value).to(self.cfg.device)
                    obs.append(value)
                
            if self.oracle_keys:
                for _, value in oracle_obs.items():
                    value = torch.tensor(value).to(self.cfg.device)
                    obs.append(value)

            # Concatenate all target data into flat tensor
            obs = torch.cat(obs, dim=-1)

            # Expect that models will generally require a batch dimension (B, D_total)
            if obs.ndim == 1: obs = obs.unsqueeze(0)

        else: obs = (camera_obs, proprio_obs, sensor_obs, oracle_obs)

        with torch.no_grad():
            try:
                actions = self.model(obs)
            except Exception as e:
                logger.warning(
                    f"Error processing model input: {e}"
                )
                raise ValueError(
                    "Ensure that config parameter 'model_use_structured_obs' "
                    "is correctly specified for the model you are using, and "
                    "you have correctly specified the observation space."
                )

        return actions