from typing import List, Tuple, Dict, Union

import numpy as np
import torch


def batch_dice(label,
               predicted):
    if isinstance(label, torch.Tensor):
        # This is, assuming the label itself is always on the CPU and not attached to a computational graph (saves time)
        label_ = torch.clone(label).numpy().squeeze()
    elif isinstance(label, np.ndarray):
        label_ = np.copy(label).squeeze()
    else:
        raise TypeError(f'Invalid data type of label array/tensor: {type(label)}')

    if isinstance(predicted, torch.Tensor):
        predicted_ = torch.clone(predicted.cpu().detach()).numpy().squeeze()
    elif isinstance(predicted, np.ndarray):
        predicted_ = np.copy(predicted).squeeze()
    else:
        raise TypeError(f'Invalid data type of predicted array/tensor: {type(label)}')




def compute_dice(label: Union[torch.Tensor, np.ndarray],
                 predicted: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Compute the dice coefficient between the predicted value and the ground truth. Both images should be 1-channel 0-255
    In this version of the method, the predicted array/tensor was already binarised.
    :param label: Per-pixel values of the ground truth.
    :param predicted: Per-pixel values of the predicted label.
    :return: The dice score.
    """

    # Ensure immutability by copying
    if isinstance(label, torch.Tensor):
        # This is, assuming the label itself is always on the CPU and not attached to a computational graph (saves time)
        label_ = torch.clone(label).numpy().squeeze()
    elif isinstance(label, np.ndarray):
        label_ = np.copy(label).squeeze()
    else:
        raise TypeError(f'Invalid data type of label array/tensor: {type(label)}')

    if isinstance(predicted, torch.Tensor):
        predicted_ = torch.clone(predicted.cpu().detach()).numpy().squeeze()
    elif isinstance(predicted, np.ndarray):
        predicted_ = np.copy(predicted).squeeze()
    else:
        raise TypeError(f'Invalid data type of predicted array/tensor: {type(label)}')

    label_[label_ > 0] = 1
    predicted_[predicted_ > 0] = 1

    intersection_length = np.sum([predicted_[label_ == 1]])
    label_length = np.sum(label_ == 1)
    predict_length = np.sum(predicted_ == 1)

    if not np.isnan(intersection_length):
        return (2 * intersection_length) / (label_length + predict_length)

    return np.array([0])