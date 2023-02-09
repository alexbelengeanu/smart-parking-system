import logging
import os
import numpy as np
from PIL import Image


class Logger:
    def __init__(self,
                 name: str,
                 log_entry: str,
                 level=logging.DEBUG):
        """
        Initialize Logger object

        Args:
            name: The name of the Logger object
            level: The logging level for the Logger object
        """
        logger = logging.getLogger(name)
        self.logger = logger
        self.logger.setLevel(level=level)
        self.log_entry = log_entry

        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)7s ' +
                                      'in %(name)s: %(message)s',
                                      datefmt='%d/%m/%Y %H:%M:%S')

        handler = logging.FileHandler(os.path.join(self.log_entry, 'logfile'))
        handler.setFormatter(formatter)
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_image(self, image, image_name):
        if not os.path.isdir(self.log_entry):
            self.logger.error(f'Log directory does not exist. Could not log the image.')
        else:
            save_path = f'{self.log_entry}/{image_name}.png'
            # Check if the image type is numpy array
            if type(image) == np.ndarray:
                image = Image.fromarray(image)
                image.save(save_path)
                self.logger.debug(f'Saved image at: {save_path}')
            elif type(image) == Image.Image:
                image.save(save_path)
                self.logger.debug(f'Saved image at: {save_path}')
            else:
                self.logger.error(f'Could not save image. Image type not supported.')
