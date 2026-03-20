"""Auxiliary utils."""

import ipaddress
import os
import secrets
import socket
import string
from math import pi, sin
from urllib.parse import urlparse

from config import RANDOM_FILENAME_LENGTH
from logger import logger


def get_random_filename():
    """Generate cryptographically secure random filename *.jpg."""
    letters = string.ascii_letters + string.digits
    name = "".join(secrets.choice(letters) for _ in range(RANDOM_FILENAME_LENGTH))
    return "".join([name, ".jpg"])


def clear_cache(files_to_delete: list):
    """Delete unnecessary image files from cache."""
    for file in files_to_delete:
        try:
            os.remove(file)
        except OSError as e:
            logger.info(f"clear_cache: could not delete {file}: {e}")
    files_to_delete.clear()


def calibrate_confidence(confidence: float) -> float:
    """Calculate calibrated confidence."""
    return (confidence + sin(confidence * pi) / 5) ** 0.7


def uri_validator(uri: str) -> bool:
    """Validate URI string. Only http/https schemes are allowed.
    Rejects requests to private, loopback, and link-local addresses (SSRF protection).

        Args:
            uri: uri string for validating.

        Returns:
            True if "uri" is a valid public http/https URL, else - False.
    """
    try:
        assert len(uri) > 0
        result = urlparse(uri)
        if result.scheme not in ("http", "https"):
            return False
        if not result.netloc:
            return False
        hostname = result.hostname
        if not hostname:
            return False
        resolved = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in resolved:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                logger.info(f"uri_validator: blocked private/reserved address {addr} for {uri}")
                return False
        return True
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
