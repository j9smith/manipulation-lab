import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

@hydra.main(config_path="../config/train", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pass

if __name__ == "__main__":
    main()