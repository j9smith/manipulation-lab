import logging
logger = logging.getLogger("ManipulationLab.DataLoader")

from manipulation_lab.scripts.dataset.wrapper import DatasetWrapper, SequentialDatasetWrapper
from manipulation_lab.scripts.dataset.reader import DatasetReader
from torch.utils.data import DataLoader
from typing import Optional
import torch
from torch.nn import Module
from functools import partial
import random

def build_dataloaders(
    cfg, 
    encoder:Optional[Module] = None,
    structured_obs:bool = False
):
    reader = DatasetReader(cfg.dataset.dataset_dirs)
    train_eps, val_eps, test_eps = _split_episodes(
        reader=reader,
        splits=cfg.dataset.splits,
        seed=cfg.dataset.seed
    )

    logger.info("Building DataLoaders")
    datasets = {}
    dataloaders = {}
    for split, episodes in zip(['train', 'val', 'test'], [train_eps, val_eps, test_eps]):
        datasets[split]= SequentialDatasetWrapper(
            dataset_dirs=cfg.dataset.dataset_dirs,
            episode_indices=episodes,
            camera_keys=cfg.dataset.camera_keys,
            action_keys=cfg.dataset.action_keys,
            sequence_length=cfg.dataset.sequence_length,
            stride=cfg.dataset.stride,
            proprio_keys=cfg.dataset.proprio_keys,
            sensor_keys=cfg.dataset.sensor_keys,
            image_encoder=encoder,
        )

        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=cfg.dataloader.batch_size,
            shuffle=(split=="train"),
            num_workers=cfg.dataloader.num_workers,
            collate_fn=partial(
                collate_fn, 
                encoder=encoder,
                structured_obs=structured_obs
            )
        )
    
    return dataloaders

def _split_episodes(reader, splits, seed):
    logger.info("Splitting episodes into train, test, and val sets.")

    # Filter out DAgger datasets for val/test splits
    clean_indices = [idx for idx, source in enumerate(reader.episode_sources) if source == "clean"]
    dagger_indices = [idx for idx, source in enumerate(reader.episode_sources) if source == "dagger"]

    episode_indices = clean_indices + dagger_indices

    random.Random(seed).shuffle(episode_indices)

    n_test = int(splits["test"] * len(clean_indices))
    n_val = int(splits["val"] * len(clean_indices))
    n_train = len(episode_indices) - n_test - n_val

    assert n_test + n_val + n_train == len(episode_indices), "Splits exceeded n episodes"

    val_ep_indices = clean_indices[:n_val]
    test_ep_indices = clean_indices[n_val: n_val + n_test]

    reserved_indices = set(val_ep_indices + test_ep_indices)

    train_ep_indices = [idx for idx in episode_indices if idx not in reserved_indices]

    logger.info(
        f"Number of episodes: Train: {len(train_ep_indices)} | "
        f"Val: {len(val_ep_indices)} | Test: {len(test_ep_indices)}"
    )

    return train_ep_indices, val_ep_indices, test_ep_indices


def collate_fn(batch, structured_obs:bool, encoder:Optional[Module] = None):
    """
    Receives the data in a batch of dictionaries of the format:
    {
        "metadata": {
            "episode_idx": int,
            "start_frame_idx": int
        },
        "actions": {
            "action_key": torch.Tensor(T, N,),
        },
        "camera": {
            "camera_key": torch.Tensor(T, C, H, W),
        },
        "proprio": {
            "proprio_key": torch.Tensor(T, N,),
        },
        "sensor": {
            "sensor_key": torch.Tensor(T, N,),
        }
    }

    And processes them into batch format.

    Parameters:
    - structured_obs: bool - If true, return a dictionary of tensors with sensor keys,
    otherwise return a flat tensor.
    - encoder: Optional[Module] - The encoder used to process the camera data.

    Returns:
    If structured_obs: A dictionary of tensors with sensor keys of shape (B, T, D),
    e.g: {
        "sensors/wrist_camera/rgb": torch.Tensor(B, T, 1),
        ...
    }
    Else: Returns a single tensor with flattened observations of shape (B, T, D_total),
    e.g. B, T, (encoded_camera_dims + proprio_dims + ...)

    """
    obs = {}
    if "camera" in batch[0].keys():
        # TODO: What if we're returning structured obs and handling obs within the model?
        if encoder is None: raise ValueError(
            "Encoder is required when using camera data. No encoder specified."
        )
        device = next(encoder.parameters()).device

        camera_keys = batch[0]["camera"].keys()
        for camera_key in camera_keys:
            # Stack camera_key images across batch
            images = torch.stack([dict["camera"][camera_key] for dict in batch])

            B, T, C, H, W = images.shape

            # Condense time dimension for encoder compatability
            images = images.view(B * T, C, H, W).to(device)

            # Batch encode the images
            encoded = encoder(images).cpu()

            # Restore time dimension
            obs[camera_key] = encoded.view(B, T, -1)

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
        # Tensors have shape B, T, ...
        obs = torch.cat(list(obs.values()), dim=-1)
        return obs, actions

