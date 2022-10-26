import gc
import logging
import os
from decimal import Decimal
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import DictConfig, open_dict
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image
from system.detection.dataset import SegmentationDataset
from system.detection.model import deeplab_v3


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
        self.best_valid_loss = float('inf')
        self.best_model: Optional[torch.nn.Module] = None
        self.best_epoch = 0

    def make_dirs(self) -> None:
        """
        Set up the local directories needed for storing various experiment results.
        :return: None
        """
        pass

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
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0)

        # Test dataset and loader
        test_images_path = os.path.join('dataset', 'segmentation', 'test', 'images')
        test_labels_path = os.path.join('dataset', 'segmentation', 'test', 'labels')

        self.test_dataset = SegmentationDataset(samples_folder=test_images_path,
                                                labels_folder=test_labels_path,
                                                device=self.device)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=1,
                                          num_workers=0)

    def make_model(self) -> None:
        """
        Instantiate and store the model for the experiment.
        :return: None.
        """
        self.model = deeplab_v3(self.device)

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

    def run_experiment(self) -> None:
        """
        Wraps all the steps of the experiment.
        :return: None.
        """
        self.setup()
