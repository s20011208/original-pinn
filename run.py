import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch

def excute(cfg):
    from src.heat.train import heat_training
    
    if cfg.task == 'heat':
        heat_training()
    else:
        raise NotImplementedError

@hydra.main(config_path="conf", config_name="config")
def run(cfg:DictConfig):
    from src.utils.glob import setup_logging, setup_board, setup_cfg
    setup_logging()
    setup_board()
    setup_cfg(cfg)
    from src.utils.glob import logger
    logger.info('\n'+OmegaConf.to_yaml(cfg))
    logger.info(f'executing task: {cfg.task}')
    excute(cfg)
    logger.info(f'logging folder: {os.getcwd()}')


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run()
   