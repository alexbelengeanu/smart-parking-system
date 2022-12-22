from typing import List

import numpy as np
import cv2 as cv
from PIL import Image


def preprocess_segmentation_map(segmentation_map, segmentation_threshold=70):
    """
    Function used to preprocess the segmentation map by applying a threshold value on the predict.
    Args:
        segmentation_map: Predict to preprocess
        segmentation_threshold: Segmentation threshold used to preprocess

    Returns:
        The preprocessed segmentation map (predict)
    """
    img = np.array(segmentation_map)
    img = cv.GaussianBlur(img, (7, 7), 0)
    img = cv.threshold(img, segmentation_threshold, 255, cv.THRESH_BINARY)[1]

    return img


def merge_images(images: List) -> Image:
    """

    Args:
        images: A list of images to be merged into a single one.

    Returns:
        a PIL.Image that consists of all the images in the list merged.
    """

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im
