import os
import random
import string
from math import pi, sin
from urllib.parse import urlparse

from config import RANDOM_FILENAME_LENGTH
from logger import logger


def get_random_filename():
    letters = string.ascii_letters + string.digits
    name = "".join(random.choice(letters) for i in range(RANDOM_FILENAME_LENGTH))
    return "".join([name, ".jpg"])


def clear_cache(files_to_delete: list):
    for file in files_to_delete:
        os.popen(f"rm {file}")
    files_to_delete.clear()


def calibrate_confidence(confidence: float) -> float:
    return (confidence + sin(confidence * pi) / 5) ** 0.7


def get_confidence_text(confidence: float) -> str:
    confidence = calibrate_confidence(confidence)
    if confidence > 0.005:
        conf_text = f"Я уверен на {round(confidence * 100)}%!"
    else:
        conf_text = ""
    return conf_text


def uri_validator(uri):
    try:
        assert len(uri) > 0
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except Exception as exception:
        logger.info(f"uri_validator: {str(exception)}")
        return False


def clip(val, min_=0, max_=1):
    return min_ if val < min_ else max_ if val > max_ else val


def log_error(text):
    logger.error(text)


def log_debug(text):
    logger.debug(text)
