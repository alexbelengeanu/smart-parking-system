import gc
import os
from typing import Optional

import time
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from hydra.utils import get_original_cwd

import mlflow
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

import torch
import torchvision
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import KMNIST
from torch.optim import Adam

from system.classification.model import CharacterClassifier
from system.backend.lib.logger import Logger


def tear_down() -> None:
    """
    Close the experiment.
    :return: None
    """
    gc.collect()
    torch.cuda.empty_cache()


class Trainer:
    def __init__(self,
                 config: DictConfig,
                 logger: Logger,
                 experiment: mlflow.entities.Experiment) -> None:
        """
            Initialize the trainer object.

            :param config: The hydra config file
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.config = config
        self.logger = logger
        self.experiment = experiment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.debug(f"Running training using device : {self.device}")

        self.train_dataset: Optional[Dataset] = None
        self.train_dataloader: Optional[DataLoader] = None

        self.test_dataset: Optional[Dataset] = None
        self.test_dataloader: Optional[DataLoader] = None

        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model: Optional[torch.nn.Module] = None

        self.n_epochs: Optional[int] = None
        self.best_model: Optional[torch.nn.Module] = None
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        self.version = None

        self.model_checkpoints_path = None
        self.num_classes = None
        self.run_id = None

    def make_dirs(self) -> None:
        """
            Set up the local directories needed for storing various experiment results.
            :return: None
        """
        working_dir = get_original_cwd()

        # Get current experiment version path
        version_path = os.path.join(working_dir, r"system/classification/version.txt")

        # Read current experiment version
        file_in = open(version_path, 'r')
        self.version = Decimal(file_in.read())
        file_in.close()

        # Create model checkpoints directory
        experiment_folder_path = os.path.join(working_dir, r"system/classification/training", f"experiment_v{self.version}")
        if not os.path.isdir(experiment_folder_path):
            os.mkdir(experiment_folder_path)

        self.model_checkpoints_path = os.path.join(experiment_folder_path, "model_checkpoints")
        if not os.path.isdir(self.model_checkpoints_path):
            os.mkdir(self.model_checkpoints_path)

        # Update experiment version for the next experiment
        file_out = open(version_path, "w")
        if str(self.version).split(".")[1] == "99":
            _version = str(float(str(self.version).split(".")[0]) + 1) + "0"
        else:
            _version = self.version + Decimal("0.01")
        file_out.write(str(_version))
        file_out.close()

        self.logger.debug("Successfully created checkpoint directory for the experiment.")

    def make_datasets(self) -> None:
        """
        Create and store the dataset and dataloader objects for the splits of our data.
        :return: None.
        """
        batch_size = self.config['batch_size']

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.ImageFolder(r'E:/GitHub/smart-parking-system/dataset/classification-small/',
                                                   transform=transform)
        self.num_classes = len(dataset.classes)
        self.logger.debug(f"Successfully loaded dataset with {self.num_classes} classes")

        # Train and test dataset and dataloaders
        train_length = int(.8 * len(dataset))
        test_length = len(dataset) - train_length
        self.train_dataset, self.test_dataset = random_split(dataset, [train_length, test_length])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def make_model(self) -> None:
        """
        Instantiate and store the model for the experiment.
        :return: None.
        """

        self.model = CharacterClassifier().to(self.device)

    def set_optimizer(self) -> None:
        """
        Instantiate and stores the optimizer of the experiment.
        :return: None.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def set_loss_function(self) -> None:
        """
        Select and instantiate the loss function based on the hyperparameter.
        :return: None.
        """
        loss_function = self.config['loss_function']
        if loss_function == "ce":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.logger.debug(f"Invalid loss function: {loss_function}.")

    def setup(self) -> None:
        """
        Wrap the setup of the experiment.
        :return: None
        """
        self.make_dirs()
        self.make_datasets()
        self.make_model()
        self.set_optimizer()
        self.set_loss_function()

        self.n_epochs = self.config['epochs']

    def train(self) -> None:
        accs = []
        losses = []
        val_accs = []
        val_losses = []
        for epoch in (range(self.n_epochs)):
            self.logger.debug(f"Epoch {epoch} | Training")
            self.model.train()
            # Get batches
            loss = 0.
            acc = 0.
            num_batches = 0
            for X_batch, y_batch in self.train_dataloader:
                num_batches += 1
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                y_pred = self.model(X_batch)

                loss_batch = self.criterion(y_pred, y_batch)
                loss += loss_batch.item()
                acc += accuracy_score(torch.argmax(y_pred.cpu(), axis=1), y_batch.cpu())

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

            acc /= num_batches
            loss /= num_batches
            losses.append(loss)
            accs.append(acc)

            # Log Accuracy and loss per epoch in MLflow
            mlflow.log_metric("Train Accuracy", acc)
            mlflow.log_metric("Train Loss", loss)

            # Validation set
            self.model.eval()
            num_batches = 0
            val_acc = 0.
            val_loss = 0.
            for X_batch, y_batch in self.test_dataloader:
                num_batches += 1
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

                y_pred = self.model(X_batch)
                val_acc += accuracy_score(torch.argmax(y_pred.cpu(), axis=1), y_batch.cpu())
                loss_batch = self.criterion(y_pred, y_batch)
                val_loss += loss_batch.item()

            val_acc /= num_batches
            val_loss /= num_batches
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            mlflow.log_metric("Validation Accuracy", val_acc)
            mlflow.log_metric("Validation Loss", val_loss)

            if val_loss > 0.4:
                self.logger.debug(
                    f"Found model with better validation loss than previous: {val_loss:.3f} < "
                    f"{self.best_valid_loss:.3f}, but bigger loss than 0.4.")
                if val_loss < self.best_valid_loss - 0.2:
                    self.logger.debug(
                        f"There is a major difference between the previous model loss and the current loss (> .2). "
                        f"Saving current model weights.")
                    self.best_valid_loss = val_loss
                    self.best_epoch = epoch
                    self.best_model = self.model

                    # Save the model weights and results
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoints_path,
                                            f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))
            else:
                if val_loss < self.best_valid_loss:
                    self.logger.debug(
                        f"Found model with better validation loss than previous: {val_loss:.3f} < "
                        f"{self.best_valid_loss:.3f}, and smaller loss than 0.4. Saving current model weights anyway.")
                    self.best_valid_loss = val_loss
                    self.best_epoch = epoch
                    self.best_model = self.model

                    # Save the model weights and results
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoints_path,
                                            f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))

    def generate_report(self) -> None:
        """
        Generate a report after the experiment has finished.
        :return: None.
        """
        pass

    def run_experiment(self) -> None:
        """
        Wraps all the steps of the experiment.
        :return: None.
        """
        self.setup()
        # Start MLflow
        RUN_NAME = f"experiment_v{self.version}"
        with mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=RUN_NAME) as run:
            # Retrieve run id
            self.run_id = run.info.run_id
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            self.logger.debug(f'Started running {RUN_NAME} with ID: {self.run_id}')
            self.logger.debug(f'Artifact URI : {mlflow.get_artifact_uri()}')
            mlflow.log_params(self.config)

            self.train()

        self.generate_report()
        tear_down()
