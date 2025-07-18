"""
Creates a client to use Octo for inference. Octo is JAX/TF-based so we can't use it in the same env.
"""
import logging
logger = logging.getLogger("ManipulationLab.OctoClient")

import pickle
import zmq
import torch
import numpy as np
import torch.nn.functional

import time

class OctoClient:
    def __init__(self, address: str, device: str = "cpu"):
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(address)
        self.device = device

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        """
        Need this to avoid complaints from modules expecting nn.Module parent class
        """
        pass

    def __call__(self, obs: tuple):
        camera_obs, proprio_obs, _ = obs
        
        # Camera data actually comes in as nparray.
        # TODO: Deal with it numpy native instead of converting to tensor
        prim_camera = torch.Tensor(camera_obs["sensors/left_shoulder_camera/rgb"])
        wrist_camera = torch.Tensor(camera_obs["sensors/wrist_camera/rgb"])
        joint_pos = proprio_obs["robot/joint_pos"]


        wrist_camera = wrist_camera.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (1, C, H, W)

        # Resize wristcam from 256x256 to 128x128
        wrist_camera = torch.nn.functional.interpolate(wrist_camera, size=(128, 128), mode="bilinear")
        
        wrist_camera = wrist_camera.squeeze(0).permute(1, 2, 0) # (1, C, H, W) -> (H, W, C)

        def _process_camera(obs: torch.Tensor):
            obs = obs.cpu()
            obs = obs.unsqueeze(0).unsqueeze(0)
            np_obs = obs.numpy()

            # Per Octo paper, images are normalised to [-1, 1]
            np_obs = np_obs.astype(np.float32) / 127.5 - 1.0
            return np_obs

        octo_obs = {}

        # All obs need leading (T, B, ...) dimensions
        octo_obs["image_primary"] = _process_camera(prim_camera)
        octo_obs["image_wrist"] = _process_camera(wrist_camera)
        octo_obs["robot"] = joint_pos.reshape(1, 1, -1)

        # Add masking that Octo complains about
        octo_obs["timestep_pad_mask"] = np.array([[True]])
        octo_obs["pad_mask_dict/timestep"] = np.array([[True]])
        octo_obs["pad_mask_dict/image_primary"] = np.array([[True]])
        octo_obs["pad_mask_dict/image_wrist"] = np.array([[True]])

        self.socket.send(pickle.dumps(octo_obs))

        msg = self.socket.recv()

        actions_np = pickle.loads(msg)
        return torch.tensor(actions_np).to(self.device).squeeze(0)

        



