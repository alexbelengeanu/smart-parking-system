import gc
import os
from typing import Optional

from decimal import Decimal
from hydra.utils import get_original_cwd

import mlflow
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score

import torch
import torchvision
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from system.classification.model import CharacterClassifier


def tear_down() -> None:
    """
    Close the experiment.
    Returns:
         None
    """
    gc.collect()
    torch.cuda.empty_cache()


class Trainer:
    def __init__(self,
                 config: DictConfig,
                 experiment: mlflow.entities.Experiment) -> None:
        """
        Initialize the trainer object.

        Args:
            config: The configuration dictionary.
            experiment: The mlflow experiment object.
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.config = config
        self.experiment = experiment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        Returns:
            None
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

    def make_datasets(self) -> None:
        """
        Create and store the dataset and dataloader objects for the splits of our data.
        Returns:
             None.
        """
        batch_size = self.config['batch_size']

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.ImageFolder(r'E:/GitHub/smart-parking-system/dataset/classification-small/',
                                                   transform=transform)
        self.num_classes = len(dataset.classes)

        # Train and test dataset and dataloaders
        train_length = int(.8 * len(dataset))
        test_length = len(dataset) - train_length
        self.train_dataset, self.test_dataset = random_split(dataset, [train_length, test_length])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def make_model(self) -> None:
        """
        Instantiate and store the model for the experiment.
        Returns:
             None.
        """

        self.model = CharacterClassifier().to(self.device)

    def set_optimizer(self) -> None:
        """
        Instantiate and stores the optimizer of the experiment.
        Returns:
             None.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config['lr'],
                                         momentum=self.config['momentum'],
                                         weight_decay=self.config['weight_decay'])

        # torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def set_loss_function(self) -> None:
        """
        Select and instantiate the loss function based on the hyperparameter.
        Returns:
             None.
        """
        loss_function = self.config['loss_function']
        if loss_function == "ce":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            print(f"Invalid loss function: {loss_function}.")

    def setup(self) -> None:
        """
        Wrap the setup of the experiment.
        Returns:
             None.
        """
        self.make_dirs()
        self.make_datasets()
        self.make_model()
        self.set_optimizer()
        self.set_loss_function()

        self.n_epochs = self.config['epochs']

    def train(self) -> None:
        """
        Function used to train the model.
        Returns:
            None.
        """
        f1s = []
        losses = []
        val_f1s = []
        val_losses = []
        for epoch in (range(self.n_epochs)):
            print(f"Epoch {epoch} | Training")
            self.model.train()
            # Get batches
            loss = 0.
            f1 = 0.
            num_batches = 0
            for X_batch, y_batch in self.train_dataloader:
                num_batches += 1
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)

                loss_batch = self.criterion(y_pred, y_batch)
                loss += loss_batch.item()
                f1 += f1_score(y_batch.cpu(), torch.argmax(y_pred.cpu(), axis=1), average="macro")

                loss_batch.backward()
                self.optimizer.step()

            f1 /= num_batches
            loss /= num_batches
            losses.append(loss)
            f1s.append(f1)

            # Log F1 Score and loss per epoch in MLflow
            mlflow.log_metric("Train F1 Score", f1)
            mlflow.log_metric("Train Loss", loss)

            print(f"Epoch {epoch} | Train F1 Score: {f1:.3f} | Train Loss: {loss:.3f}")

            # Validation set
            self.model.eval()
            num_batches = 0
            val_f1 = 0.
            val_loss = 0.
            for X_batch, y_batch in self.test_dataloader:
                num_batches += 1
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

                y_pred = self.model(X_batch)
                val_f1 += f1_score(y_batch.cpu(), torch.argmax(y_pred.cpu(), axis=1), average="macro")
                loss_batch = self.criterion(y_pred, y_batch)
                val_loss += loss_batch.item()

            val_f1 /= num_batches
            val_loss /= num_batches
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            mlflow.log_metric("Validation F1 Score", val_f1)
            mlflow.log_metric("Validation Loss", val_loss)

            print(f"Epoch {epoch} | Validation F1 Score: {val_f1:.3f} | Validation Loss: {val_loss:.3f}")

            if self.best_valid_loss > val_loss > 0.4:
                print(
                    f"Found model with better validation loss than previous: {val_loss:.3f} < "
                    f"{self.best_valid_loss:.3f}, but bigger loss than 0.4.")
                if val_loss < self.best_valid_loss - 0.2:
                    print(
                        f"There is a major difference between the previous model loss and the current loss (> .2). "
                        f"Saving current model weights.")
                    self.best_valid_loss = val_loss
                    self.best_epoch = epoch
                    self.best_model = self.model

                    # Save the model weights and results
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoints_path,
                                            f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))
            elif val_loss < 0.4:
                if val_loss < self.best_valid_loss:
                    print(
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
        Returns:
             None.
        """
        pass

    def run_experiment(self) -> None:
        """
        Wraps all the steps of the experiment.
        Returns:
             None.
        """
        self.setup()
        # Start MLflow
        RUN_NAME = f"experiment_v{self.version}"
        with mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=RUN_NAME) as run:
            # Retrieve run id
            self.run_id = run.info.run_id
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            print(f'Started running {RUN_NAME} with ID: {self.run_id}')
            print(f'Artifact URI : {mlflow.get_artifact_uri()}')
            mlflow.log_params(self.config)

            self.train()

        self.generate_report()
        tear_down()
