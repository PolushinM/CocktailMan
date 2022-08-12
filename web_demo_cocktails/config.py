import os

STATIC_FOLDER = "static"

CACHE_FOLDER = "cache"

MAX_FILE_SIZE = 30000000  # bytes

DEFAULT_SECRET_KEY = "12345"

SECRET_KEY = os.environ.get('SECRET_KEY', DEFAULT_SECRET_KEY)

DEFAULT_DEBUG = False

DEBUG = bool(os.environ.get('DEBUG', DEFAULT_DEBUG))

DEFAULT_DOCKER_HOST = "0.0.0.0"

DOCKER_HOST = os.environ.get('HOST', DEFAULT_DOCKER_HOST)

DEFAULT_DOCKER_PORT = 5000

DOCKER_PORT = int(os.environ.get('PORT', DEFAULT_DOCKER_PORT))

DEFAULT_WSGI_HOST = "0.0.0.0"

WSGI_HOST = os.environ.get('HOST', DEFAULT_WSGI_HOST)

DEFAULT_WSGI_PORT = 5000

WSGI_PORT = int(os.environ.get('PORT', DEFAULT_WSGI_PORT))

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/35.0.1916.47 Safari/537.36 '
}

MAX_MODERATED_SIZE = 640

CLASSIFIER_CONFIG_PATH = 'models/config/model_classifier.json'

INGREDIENTS_CONFIG_PATH = 'models/config/ingredients.json'

DETECTOR_CONFIG_PATH = 'models/config/model_detector.json'

CLASSIFIER_MODEL_PATH = 'models/classifier.onnx'

DETECTOR_MODEL_PATH = 'models/detector.onnx'

BLUR_MODEL_PATH = 'models/blur_model.onnx'

# CLASSIFICATION_BLUR_BBOX_EXPANSION = 1.0

VISUAL_BLUR_BBOX_EXPANSION = 0.8

# DETECTOR_BBOX_EXPANSION = 1.2044

DRAW_BBOX = "Debug"   # True, False or Debug

BBOX_LINE_THICKNESS = 1.5

BBOX_LINE_COLOR = "#0d3070"

VISUAL_BLUR_POWER = 4.0

# CLASSIFICATION_BLUR_POWER = 3.0

DETECTOR_BBOX_CONF_THRESHOLD = 0.1

CLASSIFIER_CONF_THRESHOLD = 0.35

GENERATOR_MODEL_PATH = 'models/generator.onnx'

GENERATOR_CONFIG_PATH = 'models/config/model_generator.json'

RANDOM_FILENAME_LENGTH = 10

