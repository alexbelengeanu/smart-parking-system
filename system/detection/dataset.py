import glob
import os
import cv2
import numpy as np
import torch
from typing import Tuple, List, Callable, Optional, Union

from torch import Tensor
from PIL import Image
from hydra.utils import get_original_cwd
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf


class SegmentationDataset(Dataset):
    """ Dataset used for license plate segmentation. """

    def __init__(self,
                 samples_folder: str,
                 labels_folder: str,
                 device: torch.device,
                 transforms: List[str] = None,
                 training: bool = False):
        self.device = device
        self.training = training

        working_dir = get_original_cwd()
        self.samples_path = os.path.join(working_dir, samples_folder)
        self.labels_path = os.path.join(working_dir, labels_folder)
        self.samples = list(sorted([pth for pth in os.listdir(self.samples_path)]))
        self.labels = list(sorted([pth for pth in os.listdir(self.labels_path)]))

        assert (len(self.samples) == len(self.labels))

    @staticmethod
    def resize(source: Image.Image,
               label: Image.Image
                ) -> Tuple[Image.Image, Image.Image]:
        """
        Rescale the given image toa  fixed size.

        :return:
        """

        source = source.resize((640, 320))
        label = label.resize((640, 320))

        return source, label

    @staticmethod
    def to_tensor(source: Image.Image,
                  label: Image.Image) -> Tuple[Tensor, Tensor]:
        """
        Convert a PIL Image to PyTorch tensor. The result is a tensor with values between 0. and 1.
        :param source: A source image / sample.
        :param label: A label image consisting of a binary mask of the license plate
        :return: The tuple with tensors equivalent to the images.
        """
        source = tf.to_tensor(source)
        label = tf.to_tensor(label)
        return source, label

    def __getitem__(self,
                    index: int) -> Tuple[Tensor, Tensor]:
        """
        Return the image at the specified index.
        :param index: An integer representing an index from the list of input data.
        :return: The source and the label at the index, taken through transforms.
        """
        source = Image.open(os.path.join(self.samples_path, self.samples[index])).convert('RGB')
        label = Image.open(os.path.join(self.labels_path, self.labels[index]))

        # Resize the images
        source, label = self.resize(source=source,
                                    label=label)

        if self.training:
            pass

        # Convert to tensor
        source, label = self.to_tensor(source=source,
                                       label=label)
        return source, label

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        :return: The length of the dataset.
        """
        return len(self.samples)
