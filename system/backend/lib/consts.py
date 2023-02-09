import os

# PATHS
SEGMENTATION_MODEL_PATH = r'E:\GitHub\smart-parking-system\system\detection\training\experiment_v0.02\model_checkpoints\v0.02_e59_l0.125.pt'
CLASSIFICATION_MODEL_PATH = r'E:/GitHub/smart-parking-system/system/classification/training/experiment_v0.14/model_checkpoints/v0.14_e9_l0.074.pt'
PROCESS_ON_IMAGES_PATH = r'E:/GitHub/smart-parking-system/inputs/'
RESULTS_PATH = r'E:/GitHub/smart-parking-system/results/'
RUN_ID_PATH = r'E:/GitHub/smart-parking-system/results/run_id.txt'

# Constants
RAW_MAX_SIZE = [195, 256]
CHARACTERS_MAPPING = {"0": "0",
                      "1": "1",
                      "2": "2",
                      "3": "3",
                      "4": "4",
                      "5": "5",
                      "6": "6",
                      "7": "7",
                      "8": "8",
                      "9": "9",
                      "10": "A",
                      "11": "B",
                      "12": "C",
                      "13": "D",
                      "14": "E",
                      "15": "F",
                      "16": "G",
                      "17": "H",
                      "18": "I",
                      "19": "J",
                      "20": "K",
                      "21": "L",
                      "22": "M",
                      "23": "N",
                      "24": "O",
                      "25": "P",
                      "26": "R",
                      "27": "S",
                      "28": "T",
                      "29": "U",
                      "30": "V",
                      "31": "W",
                      "32": "X",
                      "33": "Y",
                      "34": "Z"}
AZURE_CONFIG = {
    'user': 'alexbelengeanu',
    'password': os.getenv('AZURE_MYSQL_PASSWORD', default='not_found'),
    'host': 'smart-parking-system.mysql.database.azure.com',
    'database': "thesis"
}
