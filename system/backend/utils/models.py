from typing import Union

import torch

from system.detection.model import deeplab_v3
from system.classification.model import CharacterClassifier
from system.backend.lib.types import ParkingSystemModelEnum
from system.backend.lib.consts import SEGMENTATION_MODEL_PATH, CLASSIFICATION_MODEL_PATH


def initialize_model(model_type : ParkingSystemModelEnum) -> Union[deeplab_v3, CharacterClassifier]:
    """
    Initialize the model based on model type.
    Parameters:
        model_type : The type of the model.

    Returns:
        model : The model itself with the weights loaded.
    """
    if model_type == ParkingSystemModelEnum.SEGMENTATION:
        model = deeplab_v3('resnet101', 'imagenet', ['license-plate'], 'sigmoid')
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()

        if torch.cuda.is_available():
            model.to('cuda')
        return model

    elif model_type == ParkingSystemModelEnum.CLASSIFICATION:
        model = CharacterClassifier()
        checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()

        if torch.cuda.is_available():
            model.to('cuda')
        return model

    else:
        raise ValueError("Invalid model type.")
