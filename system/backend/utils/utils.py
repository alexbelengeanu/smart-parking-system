import os
from typing import List

import cv2
import numpy as np
from PIL import Image

from system.backend.lib.types import ProcessEnum
from system.backend.lib.consts import RUN_ID_PATH, RESULTS_PATH, RAW_MAX_SIZE


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a given image.
    Args:
        image: The image to apply histogram equalization to.
    Returns:
        The image after applying histogram equalization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    return equalized


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0) -> np.ndarray:
    """
    Return a sharpened version of the image, using an unsharp mask.
    https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    Args:
        image: The image to sharpen.
        kernel_size: The size of the blur kernel, should be an odd number.
        sigma: The standard deviation of the blur kernel.
        amount: The strength of the sharpening.
        threshold: The minimum difference between the original and the blur
            image that is required for the pixel to be included in the sharpened
            output.
    Returns:
        The sharpened image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def image_resize(image, width=None, height=None, inter=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize an image to a given width and height.
    https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    Args:
        image: The image to sharpen.
        width: Width of the new image
        height: Height of the new image
        inter: Interpolation method

    Returns:
        The resized image.
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def extract_segmentation_results(image, mask, resize_width=None, resize_height=256, sharpen_amount=10,
                                 crop_min_area_rect=False) -> List[np.ndarray]:
    """
    Extract the figures from the image using the segmentation mask.
    Args:
        image: The image to extract the figures from.
        mask: The segmentation mask.
        resize_width: The width of the result cropped figures.
        resize_height: The height of the result cropped figures.
        sharpen_amount: The amount of sharpening to apply to the result cropped figures.
        crop_min_area_rect: Crop the figures using the minimum area rectangle.

    Returns:

    """
    image = cv2.cvtColor(np.array(image, copy=True), cv2.COLOR_RGB2BGR)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    figures = []
    for c in contours:
        if crop_min_area_rect is True:
            # https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            angle = rect[2]
            if angle < -45:
                angle += 90

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
            cropped = cv2.getRectSubPix(image, size, center)
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = H if H > W else W
            croppedH = H if H < W else W

            result = cv2.getRectSubPix(
                cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
        else:
            x, y, w, h = cv2.boundingRect(c)
            result = image[y:y + h, x:x + w]

        result = image_resize(
            result, height=resize_height, width=resize_width)
        if sharpen_amount is not None:
            result = unsharp_mask(result, amount=sharpen_amount)
        figures.append(cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB))

    return figures


def create_results_directory(filename: str = None) -> str:
    """
    Create a new directory for the results of the current run.
    Args:
        filename: Input filename for the current run.

    Returns:
        The path of the new directory.
    """
    file_in = open(RUN_ID_PATH, 'r')
    run_id = int(file_in.read())
    file_in.close()
    if filename:
        new_run_folder_path = os.path.join(RESULTS_PATH, f"{run_id}-{ProcessEnum.ON_IMAGE.name.lower()}-{filename}")
        os.mkdir(new_run_folder_path)
    else:
        new_run_folder_path = os.path.join(RESULTS_PATH, f"{run_id}-{ProcessEnum.ON_VIDEO.name.lower()}")
        os.mkdir(new_run_folder_path)
        with open(os.path.join(new_run_folder_path, 'vehicle_id.txt'), 'w') as f:
            f.write('0')

    file_out = open(RUN_ID_PATH, "w")
    file_out.write(str(run_id + 1))
    file_out.close()

    return new_run_folder_path


def draw_bboxes(bboxes: list,
                contours: list,
                image: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on the image.
    Args:
        bboxes: Bounding boxes to draw.
        contours: Contours of the characters on which we will draw the bounding boxes.
        image: The image on which we will draw the bounding boxes.

    Returns:
        The image with the bounding boxes on it.
    """
    for idx in bboxes:
        x, y = contours[idx].T
        bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
        image = cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 1)

    return image


def add_padding(character: Image.Image) -> Image.Image:
    """
    Add padding to the character image.
    Args:
        character: Image with the character.

    Returns:
        Image with the character and padding added.
    """
    width, height = character.size
    x_axis_offset = (RAW_MAX_SIZE[0] - width) / 2
    y_axis_offset = (RAW_MAX_SIZE[1] - height) / 2

    new_width = width + int((2 * x_axis_offset))
    new_height = height + int((2 * y_axis_offset))

    padded_character = Image.new(character.mode, (new_width, new_height), 255)
    padded_character.paste(character, (int(x_axis_offset), int(y_axis_offset)))

    return padded_character
