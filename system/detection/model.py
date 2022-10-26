from torchvision.models.segmentation.deeplabv3 import DeepLabHead
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
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048,
                                   outputchannels)

    # Set CUDA device
    model.to_device(device)

    return model
