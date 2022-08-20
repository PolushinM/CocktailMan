"""Logger with custom format of logs."""

import logging
from config import DEBUG


class CustomFormatter(logging.Formatter):
    """Formatter instances are used to convert a LogRecord to text.
    Formatter need to know how a LogRecord is constructed. It
    responsible for converting a LogRecord to a string which can
    be interpreted by human."""

    grey = "\x1b[0;37m"
    white = "\x1b[0;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"

    str_format = "%(asctime)s %(message)s"

    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + str_format + reset,
        logging.INFO: white + str_format + reset,
        logging.WARNING: yellow + str_format + reset,
        logging.ERROR: red + str_format + reset,
        logging.CRITICAL: bold_red + str_format + reset
    }

    def format(self, record):
        """Format the specified record as text.
        The record's attribute dictionary is used as the operand to a
        string formatting operation which yields the returned string."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.default_time_format = '%H:%M:%S'
        formatter.default_msec_format = '%s.%03d'
        return formatter.format(record)


class CustomLogger(logging.Logger):
    """Custom logger with custom logs formatting."""

    levels = {'CRITICAL': 50, 'FATAL': 50, 'Fatal': 50, 'fatal': 50, 1: 50,
              'ERROR': 40, 'Error': 40, 'error': 40, 2: 40, False: 40,
              'WARNING': 30, 'WARN': 30, 'Warning': 30, 'warning': 30, 3: 30,
              'INFO': 20, 'Info': 20, 'info': 20, 4: 20, True: 20,
              'DEBUG': 10, 'Debug': 10, 'debug': 10, 5: 10,
              'NOTSET': 0, 0: 0}

    def __new__(cls, verbosity_level):
        custom_logger = logging.getLogger("Cocktailman")
        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(CustomFormatter())
        custom_logger.addHandler(handler)
        custom_logger.setLevel(cls.levels[verbosity_level])
        return custom_logger


logger = CustomLogger("debug" if DEBUG else logging.NOTSET)
