from manipulation_lab.scripts.utils.open_loop import OpenLoopHandler

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="../config", config_name="play_config", version_base=None)
def main(cfg: DictConfig):
    openloop_handler = OpenLoopHandler(cfg)
    openloop_handler.run_open_loop()

if __name__ == "__main__":
    main()