import gc
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
from system.backend.utils.segmentation import preprocess_segmentation_map, merge_images
from system.backend.utils.metrics import compute_dice
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
                 logger: Logger) -> None:
        """
        Initialize the trainer object.
        :param config: The hydra config file
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.debug(f"Running training using device : {self.device}")

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
        self.version = None

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

        # Create training outputs directories
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

        # Update experiment version for the next experiment
        file_out = open(version_path, "w")
        if str(self.version).split(".")[1] == "99":
            _version = str(float(str(self.version).split(".")[0]) + 1) + "0"
        else:
            _version = self.version + Decimal("0.01")
        file_out.write(str(_version))
        file_out.close()

        self.logger.debug("Succesfully created output directories for the training experiment.")

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
        self.logger.debug(
            f"Loaded train dataset with {len(self.train_dataset)} samples.")

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
        self.logger.debug(
            f"Loaded validation dataset with {len(self.validation_dataset)} samples.")

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
        self.logger.debug(
            f"Loaded test dataset with {len(self.test_dataset)} samples.")

        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)

    def make_model(self) -> None:
        """
        Instantiate and store the model for the experiment.
        :return: None.
        """

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
        # Initialize SMP Train Epoch object
        train_epoch = TrainEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            optimizer=self.optimizer,
            device=self.device,
            verbose=True)

        # Initialize SMP Validation Epoch object
        valid_epoch = ValidEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            device=self.device,
            verbose=True)

        train_logs_list, valid_logs_list = [], []

        for epoch in range(self.n_epochs):
            self.logger.debug(f"Epoch {epoch} | Training")
            train_logs = train_epoch.run(self.train_dataloader)
            valid_logs = valid_epoch.run(self.validation_dataloader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            if valid_logs["dice_loss"] > 0.1:
                self.logger.debug(
                    f"Found model with better validation loss than previous: {valid_logs['dice_loss']:.3f} < "
                    f"{self.best_valid_loss:.3f}, but bigger loss than 0.1.")
                if valid_logs["dice_loss"] < self.best_valid_loss - 0.2:
                    self.logger.debug(
                        f"There is a big difference between the previous model and the current one (> .2). Saving "
                        f"current model weigths.")
                    self.best_valid_loss = valid_logs["dice_loss"]
                    self.best_epoch = epoch
                    self.best_model = self.model

                    # Save the model weights and results
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoints_path,
                                            f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))
            else:
                self.logger.debug(
                    f"Found model with better validation loss than previous: {valid_logs['dice_loss']:.3f} < "
                    f"{self.best_valid_loss:.3f}, and smaller loss than 0.1. Saving current model weights.")
                self.best_valid_loss = valid_logs["dice_loss"]
                self.best_epoch = epoch
                self.best_model = self.model

                # Save the model weights and results
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_checkpoints_path,
                                        f'v{self.version}_e{self.best_epoch}_l{self.best_valid_loss:.3f}.pt'))

    def test(self) -> None:
        self.logger.debug(f"Running model on test dataset.")
        test_epoch = ValidEpoch(
            self.model,
            loss=DiceLoss(),
            metrics=[IoU()],
            device=self.device,
            verbose=True,
        )

        # Run model on test dataset
        test_logs = test_epoch.run(self.test_dataloader)
        self.logger.debug("Evaluation on Test Data: ")
        self.logger.debug(f"Mean IoU Score: {test_logs['iou_score']:.4f}")
        self.logger.debug(f"Mean Dice Loss: {test_logs['dice_loss']:.4f}")
        self.logger.debug(f'test_logs data : {test_logs}')

        # Save model output on test dataset
        for batch_idx, (images, labels) in enumerate(self.test_dataloader):
            # Make predict on input image
            for sample_idx, image in enumerate(images):
                prediction = self.best_model(image.unsqueeze(0).to(self.device))

                # Convert predict to PIL.Image
                predicted_image = to_pil_image(torch.squeeze(prediction).cpu())
                threshold_predicted_image = Image.fromarray(
                    preprocess_segmentation_map(predicted_image))

                # Convert label to PIL.Image
                labels_image = to_pil_image(torch.squeeze(labels[sample_idx]).cpu())

                # Convert label to PIL.Image
                input_image = to_pil_image(torch.squeeze(image).cpu())

                # Merge input image, ground truth and predict
                result_images_list = [labels_image, input_image, threshold_predicted_image]
                merged_result = merge_images(result_images_list)

                merged_result.save(os.path.join(self.test_results_path,
                                                f"Batch #{batch_idx}_Sample #{sample_idx}.png"))

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
        self.train()
        self.test()
        self.generate_report()
        tear_down()
