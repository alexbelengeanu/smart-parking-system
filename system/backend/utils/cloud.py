import mysql.connector
from mysql.connector import errorcode

from system.backend.lib.logger import Logger
from system.backend.lib.consts import AZURE_CONFIG


def get_connector(logger: Logger) -> mysql.connector:
    try:
        conn = mysql.connector.connect(**AZURE_CONFIG)
        logger.debug("Connection established")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logger.error("Something is wrong with the username or password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logger.error("Database does not exist.")
        else:
            logger.error(err)
    else:
        return conn


def check_plate_number(plate_number: str,
                       connector: mysql.connector) -> bool:

    query = f"SELECT * FROM AllowedVehicles WHERE PlateNumber = '{plate_number}'"
    cursor = connector.cursor()
    cursor.execute(query)
    return cursor.fetchone() is not None
