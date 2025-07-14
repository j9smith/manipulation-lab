import logging
logger = logging.getLogger("ManipulationLab.Train")

import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from manipulation_lab.scripts.dataset.loader import build_dataloader

import torch
import time

@hydra.main(config_path="../config/train", config_name="config", version_base=None)
def main(cfg: DictConfig):
    encoder = instantiate(cfg.dataset.image_encoder) if cfg.dataset.image_encoder is not None else None
    # TODO: Allow custom dataset
    if cfg.custom_dataset is not None: raise NotImplementedError("Custom dataset not implemented in train.py")
    if cfg.custom_dataloader is None:
        dataloader = build_dataloader(
            cfg=cfg, 
            encoder=encoder, 
            structured_obs=cfg.dataset.structured_obs
            )
    else: dataloader = instantiate(cfg.custom_dataloader)

    sample_batch = next(iter(dataloader))

    if isinstance(sample_batch, (tuple, list)):
        obs, _ = sample_batch
        input_dim = obs.shape[-1]
        cfg.model.input_dim = input_dim
    else: raise TypeError(f"Invalid batch type: {type(sample_batch)}")

    try:
        model = instantiate(cfg.model)
    except Exception as e:
        logger.critical(
            f"Failed to instantiate model:\n{e}\n"
            "Did you forget to pass model parameters via config?"
            )
        sys.exit(1)

    assert cfg.device in ["cpu", "cuda"], f"Invalid device: {cfg.device}"
    model = model.to(cfg.device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.lr
    )

    logger.info(
        f"New training run:\n"
        "======================\n"
        f"Dataset: Dataset dir: {cfg.dataset.dataset_dir}\n"
        f"Dataloader: Batch size: {cfg.dataloader.batch_size} | Shuffle: {cfg.dataloader.shuffle} | Num workers: {cfg.dataloader.num_workers}\n"
        f"Model: Input dim: {cfg.model.input_dim} | Output dim: {cfg.model.output_dim}\n"
        f"Optimiser: Learning rate: {cfg.optim.lr}\n"
        f"Train: {cfg.epochs} epochs\n"
        f"Saving weights to: {cfg.train.save_dir}/{cfg.train.save_name}.pth\n"
        "======================\n"
    )
    training_start_time = time.time()
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        for batch in dataloader:
            obs, actions = batch
            obs, actions = obs.to(cfg.device), actions.to(cfg.device)

            optimiser.zero_grad()

            pred = model(obs)
            loss = torch.nn.functional.mse_loss(pred, actions)

            loss.backward()
            optimiser.step()
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} / {cfg.epochs} | Loss: {loss.item():.4f} | Time: {epoch_duration:.2f}s")

    torch.save(model.state_dict(), f"{cfg.train.save_dir}/{cfg.train.save_name}.pth")
    logger.info(f"Training completed in {time.time() - training_start_time:.2f}s")
    logger.info(f"Weights saved to {cfg.train.save_dir}/{cfg.train.save_name}.pth")

if __name__ == "__main__":
    main()