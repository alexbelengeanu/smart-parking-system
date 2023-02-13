from typing import List, Union

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from system.detection.model import deeplab_v3
from system.backend.utils.utils import image_resize, unsharp_mask, get_first_quartile_of_areas_from_bboxes
from system.backend.lib.types import ThresholdEnum
from system.backend.lib.logger import Logger


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
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.threshold(img, segmentation_threshold, 255, cv2.THRESH_BINARY)[1]

    return img


def get_segmentation_mask(input_sample: Image.Image,
                          model: deeplab_v3,
                          device: torch.device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_sample)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    return output


def get_contours(image: np.ndarray,
                 pixel_threshold: int = 75,
                 threshold_type: ThresholdEnum = ThresholdEnum.BINARY,
                 log_images: bool = False,
                 logger: Logger = None) -> List:

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(image, kernel, iterations=1)

    if threshold_type == ThresholdEnum.TOZERO:
        threshold_type = cv2.THRESH_TOZERO
    elif threshold_type == ThresholdEnum.BINARY:
        threshold_type = cv2.THRESH_BINARY
    else:
        raise ValueError("Invalid threshold type.")

    _, threshold = cv2.threshold(img_dilation, pixel_threshold, 255, threshold_type)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if log_images is True:
        logger.log_image(threshold, "img_threshold")
        logger.log_image(img_dilation, "img_dilation")

    return contours, threshold

def get_cropped_license_plate(input_sample: Image.Image,
                              license_plate_bbox: tuple):
    cropped_plate_result = np.array(input_sample)
    cropped_plate_result = cropped_plate_result[max(license_plate_bbox[0][1], 0): min(license_plate_bbox[1][1],
                                                                                      cropped_plate_result.shape[0]),
                                                max(license_plate_bbox[0][0], 0): min(license_plate_bbox[1][0],
                                                                                      cropped_plate_result.shape[1])]

    cropped_plate_result = image_resize(cropped_plate_result, height=256)
    cropped_plate_result = unsharp_mask(cropped_plate_result, amount=50)

    return cropped_plate_result


def filter_bboxes_noise(bboxes: list,
                        contours: list,
                        image: np.ndarray):
    filtered_bboxes = []

    for idx in bboxes:
        x, y = contours[idx].T
        bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
        character = image[
                    max(bbox[0][1], 0): min(bbox[1][1], image.shape[0]),
                    max(bbox[0][0], 0): min(bbox[1][0], image.shape[1])]
        bbox_area = (max(bbox[0][1], 0) + min(bbox[1][1], image.shape[0])) * \
                    (max(bbox[0][0], 0) + min(bbox[1][0], image.shape[1]))

        _, threshold_character = cv2.threshold(character, 175, 255, cv2.THRESH_BINARY)
        values, counts = np.unique(threshold_character, return_counts=True)
        if counts[0] > 2000 and counts[1] > (counts[0] / 2) and len(counts) > 1:
            filtered_bboxes.append(bbox)

    """area_threshold = get_first_quartile_of_areas_from_bboxes(filtered_bboxes, image)

    for bbox in filtered_bboxes:
        bbox_area = (max(bbox[0][1], 0) + min(bbox[1][1], image.shape[0])) * \
                    (max(bbox[0][0], 0) + min(bbox[1][0], image.shape[1]))
        if bbox_area < area_threshold:
            filtered_bboxes.remove(bbox)"""

    return filtered_bboxes


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
