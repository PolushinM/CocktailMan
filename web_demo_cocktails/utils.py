"""Auxiliary utils."""

import os
import random
import string
from math import pi, sin
from urllib.parse import urlparse

from config import RANDOM_FILENAME_LENGTH
from logger import logger


def get_random_filename():
    """Generate random filename *.jpg."""
    letters = string.ascii_letters + string.digits
    name = "".join(random.choice(letters) for i in range(RANDOM_FILENAME_LENGTH))
    return "".join([name, ".jpg"])


def clear_cache(files_to_delete: list):
    """Delete unnecessary image files from cache."""
    for file in files_to_delete:
        os.popen(f"rm {file}")
    files_to_delete.clear()


def calibrate_confidence(confidence: float) -> float:
    """Calculate calibrated confidence."""
    return (confidence + sin(confidence * pi) / 5) ** 0.7


def uri_validator(uri: str) -> bool:
    """Validate URI string.
        Args:
            uri: uri string for validating.

        Returns:
            True if "uri" is valid uri, else - False.
    """
    try:
        assert len(uri) > 0
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except Exception as exception:
        logger.info(f"uri_validator: {str(exception)}")
        return False


def clip(val, min_=0, max_=1):
    """Clip value between min and max values."""
    return min_ if val < min_ else max_ if val > max_ else val


def log_error(text):
    """Add error text to log."""
    logger.error(text)


def log_debug(text):
    """Add debug text to log."""
    logger.debug(text)
