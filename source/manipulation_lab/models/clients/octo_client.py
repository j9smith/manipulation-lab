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

        self.step = 0
        self._prev_obs = None

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        """
        Need this to avoid complaints from modules expecting nn.Module parent class
        """
        pass

    def __call__(self, obs: tuple):
        self.step += 1
        camera_obs, proprio_obs, _, _ = obs
        
        # Camera data actually comes in as nparray.
        # TODO: Deal with it numpy native instead of converting to tensor
        prim_camera = torch.Tensor(camera_obs["sensors/scene_camera/rgb"])
        wrist_camera = torch.Tensor(camera_obs["sensors/wrist_camera/rgb"])
        joint_pos = proprio_obs["robot/joint_pos"]

        wrist_camera = wrist_camera.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (1, C, H, W)

        # Resize wristcam from 256x256 to 128x128
        wrist_camera = torch.nn.functional.interpolate(wrist_camera, size=(128, 128), mode="bilinear")
        wrist_camera = wrist_camera.squeeze(0).permute(1, 2, 0) # (1, C, H, W) -> (H, W, C)

        def _process_camera(obs: torch.Tensor):
            obs = obs.cpu().unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
            # Per Octo paper, images are normalised to [-1, 1]
            return obs / 127.5 - 1.0

        prim1 = _process_camera(prim_camera)
        wrist1 = _process_camera(wrist_camera)

        if self._prev_obs is None:
            prim0, wrist0, step0 = prim1, wrist1, self.step
        else:
            prim0, wrist0, step0 = self._prev_obs
            
        octo_obs = {
            "image_primary": np.concatenate([prim0, prim1], axis=1),
            "image_wrist": np.concatenate([wrist0, wrist1], axis=1),
            "timestep": np.array([[step0, self.step]]),
            "timestep_pad_mask": np.array([[True, True]]),
            "pad_mask_dict": {
                "timestep": np.array([[True, True]]),
                "image_primary": np.array([[True, True]]),
                "image_wrist": np.array([[True, True]]),
            },
            "task_completed": np.zeros((1, 2, 4), dtype=bool),
        }

        self._prev_obs = prim1, wrist1, self.step
        self.socket.send(pickle.dumps(octo_obs))

        msg = self.socket.recv()

        actions_np = pickle.loads(msg)
        return torch.tensor(actions_np).to(self.device).squeeze(0)

        



