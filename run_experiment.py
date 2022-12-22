import hydra
import torch
import logging
from omegaconf import DictConfig

from system.detection.train import Trainer
from system.backend.lib.logger import Logger


def run_training(config: DictConfig) -> None:
    logger = Logger(name='train', level=logging.DEBUG)
    train_experiment = Trainer(config=config, logger=logger)
    train_experiment.run_experiment()
    del train_experiment


@hydra.main(config_path='conf', config_name='config.yaml', version_base="1.2")
def main(config: DictConfig):
    torch.cuda.empty_cache()
    run_training(config)


if __name__ == "__main__":
    main()
