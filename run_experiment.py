import hydra
import torch
import mlflow
import logging
from omegaconf import DictConfig

from system.detection import Trainer
from system.backend.lib import Logger


def run_training(config: DictConfig) -> None:
    logger = Logger(name='train', level=logging.DEBUG)
    experiment_controller = mlflow.set_experiment("license-plate-detection")
    train_experiment = Trainer(config=config, logger=logger, experiment=experiment_controller)
    train_experiment.run_experiment()
    del train_experiment


@hydra.main(config_path='conf', config_name='config_detection.yaml', version_base="1.2")
def main(config: DictConfig):
    torch.cuda.empty_cache()
    run_training(config)


if __name__ == "__main__":
    main()
