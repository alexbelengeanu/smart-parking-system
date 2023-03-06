import serial

from system.backend.lib.consts import ARDUINO_PORT, ARDUINO_BAUDRATE


def initialize_serial_communication():
    """
    Function used to initialize the serial communication with the Arduino board.
    """
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE)
    return ser


def tell(serial_communication,
         message):
    """
    Function used to send a message to the Arduino board.
    Args:
        serial_communication: Serial communication object
        message: Message to send

    Returns:
        None
    """
    message = "[python-sent] " + message + '\n'
    message = message.encode() # encode n send
    serial_communication.write(message)


def hear(serial_communication):
    """
    Function used to receive a message from the Arduino board.
    Args:
        serial_communication: Serial communication object

    Returns:
        The received message as string
    """
    message = serial_communication.read_until()
    message = message.decode()
    return "[python-received] " + message
