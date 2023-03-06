from enum import Enum
import cv2


class ParkingSystemModelEnum(Enum):
    """An enumeration of all the possible model types."""
    SEGMENTATION = 1
    CLASSIFICATION = 2


class ProcessEnum(Enum):
    """An enumeration of all the possible processes types."""
    ON_IMAGE = 1
    ON_VIDEO = 2


class ThresholdEnum(Enum):
    """An enumeration of all the possible threshold types."""
    BINARY = 1
    TOZERO = 2
