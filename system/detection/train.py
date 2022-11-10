import gc
import logging
import os
from decimal import Decimal
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from decimal import Decimal
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image

from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

from system.detection.dataset import SegmentationDataset
from system.detection.model import deeplab_v3
from system.backend.utils.segmentation import preprocess_segmentation_map
from system.backend.utils.metrics import compute_dice


def tear_down() -> None:
    """
    Close the experiment.
    :return: None
    """
    gc.collect()
    torch.cuda.empty_cache()


class Train:
    def __init__(self,
                 config: DictConfig) -> None:
        """
        Instantiate the train object.
        :param config: The hydra config file
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset: Optional[SegmentationDataset] = None
        self.train_dataloader: Optional[DataLoader] = None

        self.validation_dataset: Optional[SegmentationDataset] = None
        self.validation_dataloader: Optional[DataLoader] = None

        self.test_dataset: Optional[SegmentationDataset] = None
        self.test_dataloader: Optional[DataLoader] = None

        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model: Optional[torch.nn.Module] = None

        self.n_epochs: Optional[int] = None
        self.best_model: Optional[torch.nn.Module] = None
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        self.version = 0

        self.train_results_path = None
        self.test_results_path = None
        self.model_checkpoints_path = None

    def make_dirs(self) -> None:
        """
        Set up the local directories needed for storing various experiment results.
        :return: None
        """
        working_dir = get_original_cwd()

        # Get current experiment version path
        version_path = os.path.join(working_dir, r"system/detection/version.txt")

        # Read current experiment version
        file_in = open(version_path, 'r')
        self.version = Decimal(file_in.read())
        file_in.close()

        experiment_folder_path = os.path.join(working_dir, r"system/detection/training", f"experiment_v{self.version}")
        if not os.path.isdir(experiment_folder_path):
            os.mkdir(experiment_folder_path)

        self.train_results_path = os.path.join(experiment_folder_path, "train_results")
        if not os.path.isdir(self.train_results_path):
            os.mkdir(self.train_results_path)

        self.test_results_path = os.path.join(experiment_folder_path, "test_results")
        if not os.path.isdir(self.test_results_path):
            os.mkdir(self.test_results_path)

        self.model_checkpoints_path = os.path.join(experiment_folder_path, "model_checkpoints")
        if not os.path.isdir(self.model_checkpoints_path):
            os.mkdir(self.model_checkpoints_path)

        # Update experiment version for the next use
        file_out = open(version_path, "w")
        _version = self.version + Decimal("0.01")
        file_out.write(str(_version))
        file_out.close()

    def make_datasets(self) -> None:
        """
        Create and store the dataset and dataloader objects for the splits of our data.
        :return: None.
        """
        batch_size = self.config['batch_size']
        transforms = self.config['augmentation']

        # Training dataset and dataloader
        train_images_path = os.path.join('dataset', 'segmentation', 'train', 'images')
        train_labels_path = os.path.join('dataset', 'segmentation', 'train', 'labels')

        self.train_dataset = SegmentationDataset(samples_folder=train_images_path,
                                                 labels_folder=train_labels_path,
                                                 device=self.device,
                                                 transforms=transforms,
                                                 training=True)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

        # Validation dataset and loader
        validation_images_path = os.path.join('dataset', 'segmentation', 'validation', 'images')
        validation_labels_path = os.path.join('dataset', 'segmentation', 'validation', 'labels')

        self.validation_dataset = SegmentationDataset(samples_folder=validation_images_path,
                                                      labels_folder=validation_labels_path,
                                                      device=self.device)
        self.validation_dataloader = DataLoader(self.validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0)

        # Test dataset and loader
        test_images_path = os.path.join('dataset', 'segmentation', 'test', 'images')
        test_labels_path = os.path.join('dataset', 'segmentation', 'test', 'labels')

        self.test_dataset = SegmentationDataset(samples_folder=test_images_path,
                                                labels_folder=test_labels_path,
                                                device=self.device)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=batch_size,
                                          num_workers=0)

    def make_model(self) -> None:
        """
        Instantiate and store the model for the experiment.
        :return: None.
        """
        #self.model = deeplab_v3(self.device)
        self.model = deeplab_v3('resnet101', 'imagenet', ['license-plate'], 'sigmoid')

    def set_optimizer(self) -> None:
        """
        Instantiate and stores the optimizer of the experiment.
        :return: None.
        """
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config['lr'],
                                   momentum=self.config["momentum"],
                                   weight_decay=self.config["weight_decay"])

    def set_loss_function(self) -> None:
        """
        Select and instantiate the loss function based on the hyperparameter.
        :return: None.
        """
        loss_function = self.config['loss_function']
        if loss_function == "bce":
            self.criterion = torch.nn.BCELoss()
        else:
            print(f"Invalid loss function: {loss_function}. Terminating.")

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
        """
        Train the model using the available train and validation data.
        Results and metrics are logged to Core Control.
        :return: None.
        """
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch} - Training")
            self.model.train()
            train_loss = 0.0
            train_dice = 0.0
            for index, (inputs, labels) in enumerate(self.train_dataloader):
                predicted = self.model(inputs.to(self.device))
                train_batch_loss = self.criterion(predicted, labels.to(self.device))

                self.optimizer.zero_grad()
                train_batch_loss.backward()
                self.optimizer.step()

                # Transform the predicted image into a PIL image with range 0-255
                predicted_image = to_pil_image(torch.squeeze(predicted).cpu(), "L")
                threshold_predicted = preprocess_segmentation_map(predicted_image)
                labels_image = to_pil_image(torch.squeeze(labels.cpu()), "L")

                train_batch_dice = compute_dice(np.array(labels_image),
                                                np.array(threshold_predicted))

                train_loss += train_batch_loss.item()
                train_dice += train_batch_dice.item()

                # threshold_predicted - Grayscale 0-255
                threshold_predicted_image = Image.fromarray(threshold_predicted)

                # labels_image - Grayscale 0-255
                labels_image = to_pil_image(torch.squeeze(labels.cpu()), "L")

                # input_image - Grayscale 0-255
                input_image = to_pil_image(torch.squeeze(inputs.cpu()), "L")

                input_image.save(os.path.join(self.train_results_path, f'{index}_source.png'))
                labels_image.save(os.path.join(self.train_results_path, f'{index}_label.png'))
                threshold_predicted_image.save(os.path.join(self.train_results_path, f'{index}_predict.png'))

            epoch_train_loss = train_loss / len(self.train_dataloader)
            epoch_train_dice = train_dice / len(self.train_dataloader)
            print(f"Epoch {epoch}: Train Loss {epoch_train_loss:.2f} | Train Dice {epoch_train_dice:.2f}")

            valid_loss = 0.0
            valid_dice = 0.0
            with torch.no_grad():  # reduced memory use over model.eval()
                print(f"Epoch {epoch} - Validating")
                for index, (image, labels) in enumerate(self.valid_dataloader):
                    predicted = self.model(image.to(self.device))
                    val_batch_loss = self.criterion(predicted, labels.to(self.device))

                    # Todo - this will not work for batch_size > 1
                    predicted_image = to_pil_image(torch.squeeze(predicted).cpu(), "L")
                    threshold_predicted = preprocess_segmentation_map(predicted_image)
                    labels_image = to_pil_image(torch.squeeze(labels.cpu()), "L")

                    val_batch_dice = compute_dice(np.array(labels_image),
                                                  np.array(threshold_predicted))

                    valid_loss += val_batch_loss.item()
                    valid_dice += val_batch_dice.item()

                epoch_valid_loss = valid_loss / len(self.valid_dataloader)
                epoch_valid_dice = valid_dice / len(self.valid_dataloader)
                print(f"Epoch {epoch}: Valid Loss {epoch_valid_loss:.2f} | Valid Dice {epoch_valid_dice:.2f}")

                if epoch_valid_loss < self.best_valid_loss:
                    print(f"Found model with better valid loss than previous: {epoch_valid_loss:.3f} < "
                          f"{self.best_valid_loss:.3f}")
                    self.best_valid_loss = epoch_valid_loss
                    self.best_epoch = epoch
                    self.best_model = self.model

                    # Save the model weights and results
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoints_path,
                                            f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))

    def training(self) -> None:
        train_epoch = TrainEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            optimizer=optim.SGD(self.model.parameters(),
                                   lr=self.config['lr'],
                                   momentum=self.config["momentum"],
                                   weight_decay=self.config["weight_decay"]),
            device=self.device,
            verbose=True)

        valid_epoch = ValidEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            device=self.device,
            verbose=True)

        train_logs_list, valid_logs_list = [], []

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch} - Training")
            train_logs = train_epoch.run(self.train_dataloader)
            valid_logs = valid_epoch.run(self.validation_dataloader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            if valid_logs["dice_loss"] < self.best_valid_loss:
                print(f"Found model with better valid loss than previous: {valid_logs['dice_loss']:.3f} < "
                      f"{self.best_valid_loss:.3f}")
                self.best_valid_loss = valid_logs["dice_loss"]
                self.best_epoch = epoch
                self.best_model = self.model

                # Save the model weights and results
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_checkpoints_path,
                                        f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))

    def test(self):
        """
        Evaluate the model against a holdout dataset.
        The results are logged to Core Control.
        :return: None.
        """
        test_loss = 0.0
        test_dice = 0.0

        with torch.no_grad():
            for index, (input_image, labels) in enumerate(self.test_dataloader):
                # Compute the prediction and the loss on the torch tensor prediction
                predictions = self.best_model(input_image.to(self.device))
                test_batch_loss = self.criterion(predictions, labels.to(self.device))

                # Important: For torch tensors, to_pil_image converts to 0-255 uint range by default.
                predicted_image = to_pil_image(torch.squeeze(predictions).cpu(), "L")

                # threshold_predicted - Grayscale 0-255
                threshold_predicted = Image.fromarray(
                    preprocess_segmentation_map(predicted_image))

                # labels_image - Grayscale 0-255
                labels_image = to_pil_image(torch.squeeze(labels.cpu()), "L")

                test_batch_dice = compute_dice(np.array(labels_image),
                                               np.array(threshold_predicted))

                # input_image - Grayscale 0-255
                input_image = to_pil_image(torch.squeeze(input_image.cpu()), "L")

                # Store the 3 images as output
                input_image.save(os.path.join(self.test_results_path, f'{index}_img.png'))
                labels_image.save(os.path.join(self.test_results_path, f'{index}_label.png'))
                threshold_predicted.save(os.path.join(self.test_results_path, f'{index}_pred.png'))

                test_loss += test_batch_loss.item()
                test_dice += test_batch_dice.item()

            test_loss = test_loss / len(self.test_dataloader)
            test_dice = test_dice / len(self.test_dataloader)

        print(f"Test Loss {test_loss:.2f} | Test Dice {test_dice:.2f}")

    def testing(self) -> None:
        test_epoch = ValidEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            device=self.device,
            verbose=True,
        )

        test_logs = test_epoch.run(self.test_dataloader)
        print("Evaluation on Test Data: ")
        print(f"Mean IoU Score: {test_logs['iou_score']:.4f}")
        print(f"Mean Dice Loss: {test_logs['dice_loss']:.4f}")
        print(f'test_logs data : {test_logs}')

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
        self.training()
        self.testing()
        self.generate_report()
        tear_down()

