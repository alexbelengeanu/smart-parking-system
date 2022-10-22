from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

def DeepLabV3(outputchannels=1):
    """
    Initialize DeepLabv3 Class with a custom head.
    Args:
        outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabV3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048,
                                   outputchannels)

    # Set the model in training mode
    model.train()

    return model