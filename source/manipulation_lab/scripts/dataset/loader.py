import logging
logger = logging.getLogger("ManipulationLab.DataLoader")

from manipulation_lab.scripts.dataset.wrapper import DatasetWrapper
from torch.utils.data import DataLoader
from typing import Optional
import torch
from torch.nn import Module
from functools import partial

def build_dataloader(
    cfg, 
    encoder:Optional[Module] = None,
    structured_obs:bool = False
):
    logger.info("Building DataLoader")
    dataset= DatasetWrapper(
        dataset_dir=cfg.dataset.dataset_dir,
        camera_keys=cfg.dataset.camera_keys,
        action_keys=cfg.dataset.action_keys,
        proprio_keys=cfg.dataset.get("proprio_keys", None),
        sensor_keys=cfg.dataset.get("sensor_keys", None),
        transform=cfg.dataset.get("transform", None),
        image_encoder=encoder,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=partial(
            collate_fn, 
            encoder=encoder,
            structured_obs=structured_obs
        )
    )

def collate_fn(batch, structured_obs:bool, encoder:Optional[Module] = None):
    """
    Receives the data in a batch of dictionaries of the format:
    {
        "metadata": {
            "episode_idx": int,
            "frame_idx": int
        },
        "actions": {
            "action_key": torch.Tensor(N,),
        },
        "camera": {
            "camera_key": torch.Tensor(C, H, W),
        },
        "proprio": {
            "proprio_key": torch.Tensor(N,),
        },
        "sensor": {
            "sensor_key": torch.Tensor(N,),
        }
    }

    And processes them into batch format.

    Parameters:
    - structured_obs: bool - If true, return a dictionary of tensors with sensor keys,
    otherwise return a flat tensor.
    - encoder: Optional[Module] - The encoder used to process the camera data.
    """
    obs = {}
    if "camera" in batch[0].keys():
        if encoder is None: raise ValueError(
            "Encoder is required when using camera data. No encoder specified."
        )
        device = next(encoder.parameters()).device

        camera_keys = batch[0]["camera"].keys()
        for camera_key in camera_keys:
            # Stack camera_key images across batch
            images = torch.stack([dict["camera"][camera_key] for dict in batch])

            # Batch encode the images
            encoded = encoder(images.to(device)).cpu()

            obs[camera_key] = encoded

    if "sensor" in batch[0].keys():
        # TODO: Implement sensor data handling
        # We need to consider different formats: depth, lidar, etc.
        raise NotImplementedError(
            "Sensor data not supported yet. "
            "Sensor data should be handled in scripts.dataset.loader and scripts.dataset.wrapper. "
        )

    if "proprio" in batch[0].keys():
        proprio_keys = batch[0]["proprio"].keys()
        for proprio_key in proprio_keys:
            obs[proprio_key] = (torch.stack([dict["proprio"][proprio_key] for dict in batch]))

    action_keys = batch[0]["actions"].keys()
    action_data = []
    for action_key in action_keys:
        action_data.append(torch.stack([dict["actions"][action_key] for dict in batch]))

    actions = torch.cat(action_data, dim=-1)

    if structured_obs:
        return obs, actions
    else:
        return torch.cat(list(obs.values()), dim=-1), actions

