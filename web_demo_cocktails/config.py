"""Configuration file for Cocktailman application."""

from os import getenv

# General settings

DEFAULT_DEBUG = 'False'

DEBUG = getenv('DEBUG', DEFAULT_DEBUG) in {"True", "1", 'true', 'TRUE'}

# Files

STATIC_FOLDER = "static"

CACHE_FOLDER = "cache"

MAX_IMAGE_FILE_SIZE = 30000000  # bytes

MAX_IMAGE_MODERATED_SIZE = 640  # pixels

RANDOM_FILENAME_LENGTH = 10

JPEG_QUALITY = 80

# Network

DEFAULT_DOCKER_HOST = "127.0.0.1"

DOCKER_HOST = getenv('HOST', DEFAULT_DOCKER_HOST)

DEFAULT_DOCKER_PORT = 8000

DOCKER_PORT = int(getenv('PORT', DEFAULT_DOCKER_PORT))

DEFAULT_WSGI_HOST = "0.0.0.0"

WSGI_HOST = getenv('HOST', DEFAULT_WSGI_HOST)

DEFAULT_WSGI_PORT = '5000'

WSGI_PORT = int(getenv('PORT', DEFAULT_WSGI_PORT))

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/35.0.1916.47 Safari/537.36 '
}

# Security

DEFAULT_SECRET_KEY = "12345"

SECRET_KEY = getenv('SECRET_KEY', DEFAULT_SECRET_KEY)

# Models

INGREDIENTS_CONFIG_PATH = 'models/config/ingredients.json'

CLASSIFIER_MODEL_PATH = 'models/classifier.onnx'

CLASSIFIER_CONFIG_PATH = 'models/config/model_classifier.json'

CLASSIFIER_CONF_THRESHOLD = 0.35

DETECTOR_CONFIG_PATH = 'models/config/model_detector.json'

DETECTOR_MODEL_PATH = 'models/detector.onnx'

DETECTOR_BBOX_CONF_THRESHOLD = 0.1

BLUR_MODEL_PATH = 'models/blur_model.onnx'

VISUAL_BLUR_BBOX_EXPANSION = 0.8

VISUAL_BLUR_POWER = 4.0

DRAW_BBOX = "Debug"   # True, False or Debug

BBOX_LINE_THICKNESS = 1.5

BBOX_LINE_COLOR = "#0d3070"

GENERATOR_MODEL_PATH = 'models/generator.onnx'

GENERATOR_CONFIG_PATH = 'models/config/model_generator.json'
