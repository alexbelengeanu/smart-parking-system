'''from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def deeplab_v3(device,
               outputchannels=1):
    """
    Initialize DeepLabv3 Class with a custom head.
    Args:
        device: The CUDA device on which the model should run (GPU/CPU).
        outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabV3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(weights=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048,
                                   outputchannels)

    # Set CUDA device
    model.to(device)

    return model'''
from typing import List

import segmentation_models_pytorch as smp


def deeplab_v3(encoder: str,
               weights: str,
               classes: List[str],
               activation: str):
    """
    Initialize model for training.
    Args:

    Returns:
        model: Returns the model.
    """
    model = smp.DeepLabV3(
        encoder_name=encoder,
        encoder_weights=weights,
        classes=len(classes),
        activation=activation,
    )

    return model
