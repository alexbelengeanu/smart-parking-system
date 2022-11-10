import numpy as np
import cv2 as cv


def preprocess_segmentation_map(segmentation_map, segmentation_threshold = 70):

    img = np.array(segmentation_map)
    img = cv.GaussianBlur(img, (7, 7), 0)
    img = cv.threshold(img, segmentation_threshold, 255, cv.THRESH_BINARY)[1]

    return img
