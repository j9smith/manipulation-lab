import logging
logger = logging.getLogger("ManipulationLab.Train")

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from manipulation_lab.scripts.dataset.loader import build_dataloaders
import numpy as np

import torch
import torch.multiprocessing as mp
import time

import wandb

@hydra.main(config_path="../config/", config_name="train_config", version_base=None)
def main(cfg: DictConfig):
    config= OmegaConf.to_container(
            cfg, 
            resolve=True, 
            throw_on_missing=True
    )
    assert isinstance(config, dict), "wandb requires config to be a dict."

    wandb_run_name = input("Enter a save name for this W&B run: ").strip()

    run = wandb.init(
        project="Manipulation Lab",
        name=wandb_run_name,
        config=config
    )
    save_path = os.path.join(cfg.train.save_dir, f"{cfg.train.save_name}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        response = input(f"Weights file already exists at {save_path}. Overwrite? [y/n]: ").strip()
        if response not in ('y', 'Y', 'yes'):
            logger.info(f"Aborting. Please specify an alternate filename.")
            sys.exit(1)

    encoder = instantiate(cfg.encoder) if cfg.encoder is not None else None
    
    # Set torch multiprocessing start method to spawn if using cuda to avoid errors with fork
    if cfg.device == "cuda": mp.set_start_method("spawn", force=True)

    # ----- Build dataloaders -----
    # TODO: Allow custom dataset
    if cfg.custom_dataset is not None: raise NotImplementedError("Custom dataset not implemented in train.py")
    if cfg.custom_dataloader is None:
        loaders = build_dataloaders(
            cfg=cfg, 
            encoder=encoder, 
            structured_obs=cfg.dataset.structured_obs
            )
    else: dataloader = instantiate(cfg.custom_dataloader)

    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    # ----- Load model -----
    if cfg.model.input_dim is None:
        logger.info("Sampling dataloader to get input dims ...")
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (tuple, list)):
            obs, _ = sample_batch
            input_dim = obs.shape[-1]
            cfg.model.input_dim = input_dim
        else: raise TypeError(f"Invalid batch type: {type(sample_batch)}")

    logger.info("Loading model...")
    try:
        model = instantiate(cfg.model)

        if cfg.use_pretrained_weights == True:
            logger.info(f"Loading model weights from {cfg.pretrained_weights_path}")
            model.load_state_dict(
                torch.load(cfg.pretrained_weights_path, map_location=cfg.device, weights_only=True)
            )

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
        f"Model:\n{model}\n"
        f"Encoder:\n{encoder}\n"
        f"Dataset: Dataset dirs: {cfg.dataset.dataset_dirs}\n"
        f"Dataloader: Batch size: {cfg.dataloader.batch_size} | Num workers: {cfg.dataloader.num_workers}\n"
        f"Model: Input dim: {cfg.model.input_dim} | Output dim: {cfg.model.output_dim}\n"
        f"Optimiser: Learning rate: {cfg.optim.lr}\n"
        f"Train: {cfg.epochs} epochs\n"
        f"Saving weights to: {cfg.train.save_dir}/{cfg.train.save_name}.pth\n"
        "======================\n"
    )

    if cfg.pretrained_weights_path is not None and cfg.use_pretrained_weights == False:
            logger.warning(
                "Model weights specified but use_pretrained_weights is false. "
                "Training is from scratch. Ensure this behaviour is intentional."
            )

    training_start_time = time.time()
    try:
        for epoch in range(cfg.epochs):
            epoch_start_time = time.time()

            # ------ Training Pass -------
            model.train()
            total_train_loss, train_steps = 0.0, 0
            for obs, actions in train_loader:
                # We want to predict the action given frame history, so take last 
                # action in sequence as target
                obs, actions = obs.to(cfg.device), actions[:, -1, :].to(cfg.device)

                optimiser.zero_grad()

                pred = model(obs)
                loss = torch.nn.functional.mse_loss(pred, actions)

                loss.backward()
                optimiser.step()

                total_train_loss += loss.item()
                train_steps += 1

            avg_train_loss = total_train_loss / train_steps

            # ------ Validation Pass ------
            model.eval()
            with torch.no_grad():
                total_val_loss, val_steps = 0.0, 0
                for obs, actions in val_loader:
                    obs, actions = obs.to(cfg.device), actions[:, -1, :].to(cfg.device)
                    pred = model(obs)
                    loss = torch.nn.functional.mse_loss(pred, actions)
                    total_val_loss += loss.item()
                    val_steps += 1

                avg_val_loss = total_val_loss / val_steps
            
            # ------ Logging ------
            epoch_duration = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1} / {cfg.epochs} | Train. Loss: {avg_train_loss:.6f} | "
                f"Val. Loss: {avg_val_loss:.6f} | Time: {epoch_duration:.2f}s |"
                )

            wandb.log({
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Epoch Duration (s)": epoch_duration
            })

    except KeyboardInterrupt:
        logger.info(
            "Training interrupted. Saving progress before exiting ..."
        )
    
    finally:
        # ------ Post-training ------
        torch.save(model.state_dict(), save_path)
        logger.info(f"Training completed in {time.time() - training_start_time:.2f}s")
        logger.info(f"Weights saved to {cfg.train.save_dir}/{cfg.train.save_name}.pth")

        # ------ Test Evaluation -------
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for obs, actions in test_loader:
                obs = obs.to(cfg.device)
                pred = model(obs).cpu().numpy()
                predictions.append(pred)
                actuals.append(actions[:, -1, :].numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        mse_per_dimension = np.mean((predictions - actuals) ** 2, axis=0)
        average_mse = mse_per_dimension.mean()

        action_dims = predictions.shape[1]
        time_steps = list(range(predictions.shape[0]))

        for dim in range(action_dims):
            wandb.log({
                f"Predicted Action vs Actual - Dim {dim}": wandb.plot.line_series(
                    xs=time_steps,
                    ys=[actuals[:, dim], predictions[:, dim]],
                    keys=["Actual", "Predicted"],
                    title=f"Action Dimension {dim}",
                    xname="Time",
                )
            })

        wandb.finish()

if __name__ == "__main__":
    main()