import hydra
import torch
import mlflow
import logging
from omegaconf import DictConfig

from system.classification.train import Trainer
#from system.detection.train import Trainer


def run_training(config: DictConfig) -> None:
    #experiment_controller = mlflow.set_experiment("license-plate-detection")
    experiment_controller = mlflow.set_experiment("character-classification")
    train_experiment = Trainer(config=config, experiment=experiment_controller)
    train_experiment.run_experiment()
    del train_experiment


@hydra.main(config_path='conf', config_name='config_classification.yaml', version_base="1.2")
def main(config: DictConfig):
    torch.cuda.empty_cache()
    run_training(config)


if __name__ == "__main__":
    main()
