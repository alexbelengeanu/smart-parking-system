from typing import List, Tuple
import os
import torch
import matplotlib.pyplot as plt
import cv2
import logging
from PIL import Image
from torchvision import transforms
import numpy as np
import argparse

from system.detection.model import deeplab_v3
from system.classification.model import CharacterClassifier
from system.backend.utils.utils import create_results_directory, histogram_equalization, draw_bboxes, add_padding
from system.backend.utils.segmentation import get_segmentation_mask, get_contours, get_cropped_license_plate, \
    filter_bboxes_noise
from system.backend.utils.models import initialize_model
from system.backend.utils.cloud import get_connector, check_plate_number
from system.backend.lib.logger import Logger
from system.backend.lib.types import ParkingSystemModelEnum, ProcessEnum
from system.backend.lib.consts import SEGMENTATION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, PROCESS_ON_IMAGES_PATH, \
    CHARACTERS_MAPPING


def license_plate_detection(input_sample: Image.Image,
                            model: deeplab_v3,
                            device: torch.device,
                            logger: Logger) -> np.ndarray:

    # Get license plate segmentation mask
    output = get_segmentation_mask(input_sample=input_sample,
                                   model=model,
                                   device=device)

    # Convert the output from tensor to grayscale numpy array
    mask = torch.squeeze(output).cpu().numpy() > 0.7
    mask_grayscale = np.array(Image.fromarray(mask).convert("L"))
    logger.log_image(image=mask_grayscale,
                     image_name="segmentation_mask")

    # Find the biggest contour in the mask as it represents the closest license plate in the input image to the camera.
    mask_contours, _ = get_contours(image=mask_grayscale)
    areas = [cv2.contourArea(cnt) for cnt in mask_contours]
    max_area_idx = np.argmax(areas)
    x, y = mask_contours[max_area_idx].T
    license_plate_bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))

    # Crop the license plate from the input image based on bounding box + resize and sharpen
    cropped_license_plate = get_cropped_license_plate(input_sample=input_sample,
                                                      license_plate_bbox=license_plate_bbox)
    logger.log_image(image=cropped_license_plate,
                     image_name="cropped_license_plate",)

    return cropped_license_plate


def character_segmentation(cropped_license_plate: np.ndarray,
                           logger: Logger):

    histogram_eq_image = histogram_equalization(cropped_license_plate)
    logger.log_image(image=histogram_eq_image,
                     image_name="histogram_eq_image")

    all_contours, threshold = get_contours(image=histogram_eq_image,
                                           pixel_threshold=150,
                                           log_images=True,
                                           logger=logger)

    areas = [cv2.contourArea(cnt) for cnt in all_contours]
    max_area_idx = np.argmax(areas)
    x, y = all_contours[max_area_idx].T
    new_plate_bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))

    cropped_plate_without_noise = threshold[max(new_plate_bbox[0][1], 0): min(new_plate_bbox[1][1],
                                                                              threshold.shape[0]),
                                            max(new_plate_bbox[0][0], 0): min(new_plate_bbox[1][0],
                                                                              threshold.shape[1])]

    # This will be used just for visualization purposes and saved as metadata in the results' folder.
    cropped_plate_without_noise_rgb = cv2.cvtColor(cropped_plate_without_noise, cv2.COLOR_GRAY2RGB)

    logger.log_image(image=cropped_plate_without_noise,
                     image_name="cropped_plate_without_noise")

    contours, _ = cv2.findContours(cropped_plate_without_noise, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_area_idx = np.argmax(areas)
    character_bboxes = [areas.index(x) for x in sorted(areas, reverse=True)[:14]]
    character_bboxes.remove(max_area_idx)

    logger.debug(f'Got characters bounding boxes : {character_bboxes}')

    drew_bboxes = draw_bboxes(bboxes=character_bboxes,
                              contours=contours,
                              image=cropped_plate_without_noise_rgb)

    logger.log_image(image=drew_bboxes,
                     image_name="drew_bboxes")

    filtered_character_bboxes = filter_bboxes_noise(bboxes=character_bboxes,
                                                    contours=contours,
                                                    image=cropped_plate_without_noise)

    filtered_character_bboxes = sorted(filtered_character_bboxes, key=lambda x: x[0][0])
    logger.info(f'Got filtered characters bounding boxes : {filtered_character_bboxes}')

    for idx, bbox in enumerate(filtered_character_bboxes):
        logger.log_image(image=cropped_plate_without_noise[max(bbox[0][1], 0): min(bbox[1][1],
                                                                                   cropped_plate_without_noise.shape[0]),
                                                           max(bbox[0][0], 0): min(bbox[1][0],
                                                                                   cropped_plate_without_noise.shape[1])],
                         image_name=f"character_{idx}")

    return filtered_character_bboxes, cropped_plate_without_noise


def character_classification(model: CharacterClassifier,
                             filtered_character_bboxes: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                             cropped_plate_without_noise: np.ndarray,
                             logger: Logger) -> List[torch.Tensor]:
    """
    Classify the cropped characters in the license plate using the trained model.
    :param model: The trained model.
    :param filtered_character_bboxes: The bounding boxes of the characters in the license plate.
    :param cropped_plate_without_noise: The cropped license plate without noise.
    :return: The predicted characters.
    """

    license_plate_predicts_list = []
    for idx, bbox in enumerate(filtered_character_bboxes):
        character = Image.fromarray(
            cropped_plate_without_noise[max(bbox[0][1], 0): min(bbox[1][1], cropped_plate_without_noise.shape[0]),
                                        max(bbox[0][0], 0): min(bbox[1][0], cropped_plate_without_noise.shape[1])])

        padded_character = add_padding(character)
        logger.log_image(image=padded_character,
                         image_name=f"padded_character_{idx}")

        transform = transforms.Compose([transforms.ToTensor()])

        padded_character = transform(padded_character)
        padded_character = padded_character.unsqueeze(0)
        prediction = model(padded_character.cuda())

        license_plate_predicts_list.append(torch.argmax(prediction.cpu(), axis=1))

    return license_plate_predicts_list


def license_plate_reconstruction_as_string(character_classification_results: List[torch.Tensor]) -> str:
    license_plate_as_string = ""
    for character in character_classification_results:
        license_plate_as_string += (CHARACTERS_MAPPING[str(int(character[0]))])

    return license_plate_as_string


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Filename of the image to process.")
    args = parser.parse_args()

    if os.path.isfile(os.path.join(PROCESS_ON_IMAGES_PATH, args.filename)):
        results_folder_path = create_results_directory(filename=args.filename,
                                                       process_type=ProcessEnum.ON_IMAGE, )
        logger = Logger(name='process_on_image', log_entry=results_folder_path , level=logging.DEBUG)
        logger.debug(f"Successfully created results directory at: {logger.log_entry}")
    else:
        raise FileNotFoundError("File not found.")

    logger.info("Starting process on single image input.")

    model_segmentation = initialize_model(ParkingSystemModelEnum.SEGMENTATION, SEGMENTATION_MODEL_PATH)
    model_classification = initialize_model(ParkingSystemModelEnum.CLASSIFICATION, CLASSIFICATION_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Initialized both models and device.")

    connector = get_connector(logger=logger)
    logger.debug("Initialized connection to Azure Database.")

    # Read input sample and resize to (640, 320) - the model's input size.
    input_sample = Image.open(os.path.join(PROCESS_ON_IMAGES_PATH, args.filename)).convert('RGB').resize((640, 320))
    logger.log_image(image=input_sample,
                     image_name="original_image")

    cropped_license_plate = license_plate_detection(input_sample=input_sample,
                                                    model=model_segmentation,
                                                    device=device,
                                                    logger=logger)

    filtered_character_bboxes, cropped_plate_without_noise = character_segmentation(
        cropped_license_plate=cropped_license_plate,
        logger=logger)

    character_classification_results = character_classification(model=model_classification,
                                                                filtered_character_bboxes=filtered_character_bboxes,
                                                                cropped_plate_without_noise=cropped_plate_without_noise,
                                                                logger=logger)
    logger.debug(f"Predicted {len(character_classification_results)} characters.")

    license_plate_as_string = license_plate_reconstruction_as_string(
        character_classification_results=character_classification_results)

    was_found = check_plate_number(plate_number=license_plate_as_string,
                                   connector=connector)

    with open(os.path.join(results_folder_path, "output.txt"), 'w') as f:
        if was_found:
            f.write(f'License plate {license_plate_as_string} was found in the database with allowed vehicles.')
        else:
            f.write(f'License plate {license_plate_as_string} was not found in the database with allowed vehicles.')

    logger.info("Finished process on single image input.")


if __name__ == "__main__":
    main()
