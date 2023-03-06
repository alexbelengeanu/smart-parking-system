import mysql.connector
from mysql.connector import errorcode

from system.backend.lib.logger import Logger
from system.backend.lib.consts import AZURE_CONFIG


def get_connector(logger: Logger = None) -> mysql.connector:
    """
    Function used to get a connection to the database.
    Args:
        logger: Logger used to log the connection status.

    Returns:
        The connector to the database.
    """
    try:
        conn = mysql.connector.connect(**AZURE_CONFIG)
        if logger:
            logger.debug("Connection established")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            if logger:
                logger.error("Something is wrong with the username or password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            if logger:
                logger.error("Database does not exist.")
        else:
            if logger:
                logger.error(err)
    else:
        return conn


def check_plate_number(plate_number: str,
                       connector: mysql.connector) -> bool:
    """
    Function used to check if a plate number is allowed to enter the parking lot, after searching it in the database.
    Args:
        plate_number: Plate number to check.
        connector: Connector to the database.

    Returns:
        True if the plate number is found in the database, False otherwise.
    """

    query = f"SELECT * FROM AllowedVehicles WHERE PlateNumber = '{plate_number}'"
    cursor = connector.cursor()
    cursor.execute(query)
    return cursor.fetchone() is not None
