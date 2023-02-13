import serial

from system.backend.lib.consts import ARDUINO_PORT, ARDUINO_BAUDRATE


def initialize_serial_communication():
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE)
    return ser


def tell(serial_communication,
         message):
    message = "[python-sent] " + message + '\n'
    message = message.encode() # encode n send
    serial_communication.write(message)


def hear(serial_communication):
    message = serial_communication.read_until()
    message = message.decode()
    return "[python-received] " + message
