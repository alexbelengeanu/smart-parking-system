import os
import cv2
import numpy as np

from system.backend.lib.types import ProcessEnum
from system.backend.lib.consts import RUN_ID_PATH, RESULTS_PATH


def histogram_equalization(image: np.ndarray):
    """
    Apply histogram equalization to a given image.
    :param image: An image.
    :return: The image after the histogram equalization as numpy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    return equalized


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    # https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def image_resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):
    # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
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


def extract_segmentation_results(image, mask, resize_width=None, resize_height=256, sharpen_amount=10, crop_min_area_rect=False):
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

            center = ((x1+x2)/2, (y1+y2)/2)
            size = (x2-x1, y2-y1)
            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
            cropped = cv2.getRectSubPix(image, size, center)
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = H if H > W else W
            croppedH = H if H < W else W
            
            result = cv2.getRectSubPix(
                cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
        else:
            x,y,w,h = cv2.boundingRect(c)
            result = image[y:y+h, x:x+w]

        result = image_resize(
            result, height=resize_height, width=resize_width)
        if sharpen_amount is not None:
            result = unsharp_mask(result, amount=sharpen_amount)
        figures.append(cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB))

    return figures


def create_results_directory(filename: str,
                             process_type: ProcessEnum):
    file_in = open(RUN_ID_PATH, 'r')
    run_id = int(file_in.read())
    file_in.close()

    new_run_folder_path = os.path.join(RESULTS_PATH,  f"{run_id}-{process_type.name.lower()}-{filename}")
    if not os.path.isdir(new_run_folder_path):
        os.mkdir(new_run_folder_path)

    file_out = open(RUN_ID_PATH, "w")
    file_out.write(str(run_id + 1))
    file_out.close()

    return new_run_folder_path


def draw_bboxes(bboxes: list,
                contours: list,
                image: np.ndarray) -> np.ndarray:
    for idx in bboxes:
        x, y = contours[idx].T
        bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
        image = cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 1)

    return image
