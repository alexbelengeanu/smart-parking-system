from typing import List

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import segmentation_models_pytorch as smp


def deeplab_v3(encoder: str,
               weights: str,
               classes: List[str],
               activation: str):
    """
    Initialize model for training.
    Args:
        encoder: Encoder name.
        weights: Encoder weights.
        classes: List of classes in the dataset.
        activation: Activation function used.

    Returns:
        Model for training or inference.
    """
    model = smp.DeepLabV3(
        encoder_name=encoder,
        encoder_weights=weights,
        classes=len(classes),
        activation=activation,
    )

    return model
