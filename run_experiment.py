import hydra
import torch

from omegaconf import DictConfig
from system.detection.train import Train


def run_training(config: DictConfig) -> None:
    train_experiment = Train(config=config)
    train_experiment.run_experiment()
    del train_experiment


@hydra.main(config_path='conf', config_name='config.yaml', version_base="1.2")
def main(config: DictConfig):
    torch.cuda.empty_cache()
    run_training(config)


if __name__ == "__main__":
    main()
