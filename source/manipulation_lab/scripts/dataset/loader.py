import logging
logger = logging.getLogger("ManipulationLab.DataLoader")

from manipulation_lab.scripts.dataset.wrapper import DatasetWrapper
from torch.utils.data import DataLoader
from typing import Optional
from torch.nn import Module

def build_dataloader(cfg, encoder:Optional[Module] = None):
    logger.info("Building DataLoader")
    dataset= DatasetWrapper(
        dataset_dir=cfg.dataset.dataset_dir,
        camera_keys=cfg.dataset.camera_keys,
        action_keys=cfg.dataset.action_keys,
        proprio_keys=cfg.dataset.get("proprio_keys", None),
        sensor_keys=cfg.dataset.get("sensor_keys", None),
        transform=cfg.dataset.get("transform", None),
        image_encoder=encoder,
        structured_obs=cfg.dataset.structured_obs
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
    )