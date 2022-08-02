import os
import ssl

import urllib.request

from werkzeug.utils import secure_filename
from PIL import Image

from config import (CACHE_FOLDER, DEBUG, CLASSIFIER_CONF_THRESHOLD, MAX_FILE_SIZE, CLASSIFIER_CONFIG_PATH,
                    INGREDIENTS_CONFIG_PATH, CLASSIFIER_MODEL_PATH, REQUEST_HEADERS, DETECTOR_MODEL_PATH,
                    DETECTOR_CONFIG_PATH, CLASSIFICATION_BLUR_BBOX_EXPANSION, DETECTOR_BBOX_CONF_THRESHOLD,
                    MAX_MODERATED_SIZE, VISUAL_BLUR_POWER, BLUR_MODEL_PATH, DETECTOR_BBOX_EXPANSION,
                    VISUAL_BLUR_BBOX_EXPANSION, CLASSIFICATION_BLUR_POWER, DRAW_BBOX, BBOX_LINE_THICKNESS,
                    BBOX_LINE_COLOR)

from utils import get_random_filename, clear_cache
from models.models import ImageProcessor

files_to_delete = []

image_processor = ImageProcessor(classifier_model_path=CLASSIFIER_MODEL_PATH,
                                 classifier_config_path=CLASSIFIER_CONFIG_PATH,
                                 ingredients_config_path=INGREDIENTS_CONFIG_PATH,
                                 detector_model_path=DETECTOR_MODEL_PATH,
                                 detector_config_path=DETECTOR_CONFIG_PATH,
                                 detector_bbox_expansion=DETECTOR_BBOX_EXPANSION,
                                 detector_bbox_conf_threshold=DETECTOR_BBOX_CONF_THRESHOLD,
                                 classification_blur_bbox_expansion=CLASSIFICATION_BLUR_BBOX_EXPANSION,
                                 classification_blur_power=CLASSIFICATION_BLUR_POWER,
                                 blur_model_path=BLUR_MODEL_PATH,
                                 debug=DEBUG)

INGREDIENTS_TEXT = image_processor.ingredients_text

# Allow using unverified SSL for image downloading
ssl._create_default_https_context = ssl._create_unverified_context


def generate_recipe(ingredients: list) -> str:
    recipe = "Не могу найти напиток на изображении."
    if len(ingredients) > 0:
        if len(ingredients) > 1:
            recipe = ", ".join(ingredients[:-1])
            recipe = "".join(["Попробуем добавить ", recipe, " и ", ingredients[-1], "."])
        else:
            recipe = "".join(["Добавим только ", ingredients[-1], "."])
    return recipe


def moderate_size(path, max_size):
    with Image.open(path).convert("RGB") as image:
        width, height = image.size
        if (width > max_size) or (height > max_size):
            if width >= height:
                w = max_size
                h = round(max_size * height / width)
            else:
                h = max_size
                w = round(max_size * width / height)
            image = image.resize((w, h))
            image.save(path, "JPEG")


def predict(src, src_type: str) -> tuple[str, float, tuple[float, float, float, float], str]:
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)

    if src_type == "url":
        req = urllib.request.Request(src, data=None, headers=REQUEST_HEADERS)
        file = open(full_filename, "wb")
        try:
            file.write(urllib.request.urlopen(req).read(MAX_FILE_SIZE))
        except Exception as e_urllib:
            if DEBUG:
                print("urllib", e_urllib)
        finally:
            file.close()
    if src_type == "file":
        src.save(full_filename)

    moderate_size(full_filename, MAX_MODERATED_SIZE)

    ingredients, confidence, bbox = image_processor.predict(path=full_filename, threshold=CLASSIFIER_CONF_THRESHOLD)

    files_to_delete.append(full_filename)

    return generate_recipe(ingredients), confidence, bbox, filename


def blur_bounding_box(path: str, bbox: tuple[float, float, float, float]):
    image_processor.blur_bounding_box(path=path,
                                      bbox=bbox,
                                      power=VISUAL_BLUR_POWER,
                                      expansion=VISUAL_BLUR_BBOX_EXPANSION)

    if (DRAW_BBOX == "True") or (DRAW_BBOX == "Debug" and DEBUG):
        image_processor.detector.draw_bounding_box(path=path,
                                                   b_box=bbox,
                                                   thickness=BBOX_LINE_THICKNESS,
                                                   color=BBOX_LINE_COLOR)
