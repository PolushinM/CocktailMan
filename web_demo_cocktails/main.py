import os
import ssl

import urllib.request
from typing import Union

import numpy as np
from werkzeug.utils import secure_filename


from config import (CACHE_FOLDER, DEBUG, CLASSIFIER_CONF_THRESHOLD, MAX_IMAGE_FILE_SIZE, CLASSIFIER_CONFIG_PATH,
                    INGREDIENTS_CONFIG_PATH, CLASSIFIER_MODEL_PATH, REQUEST_HEADERS, DETECTOR_MODEL_PATH,
                    DETECTOR_CONFIG_PATH, DETECTOR_BBOX_CONF_THRESHOLD, MAX_IMAGE_MODERATED_SIZE, VISUAL_BLUR_POWER,
                    BLUR_MODEL_PATH, VISUAL_BLUR_BBOX_EXPANSION, DRAW_BBOX, BBOX_LINE_THICKNESS, BBOX_LINE_COLOR,
                    GENERATOR_MODEL_PATH, GENERATOR_CONFIG_PATH)

from utils import get_random_filename, clear_cache
from models.models import ImageProcessor

files_to_delete = []

image_processor = ImageProcessor(max_moderated_size=MAX_IMAGE_MODERATED_SIZE,
                                 classifier_model_path=CLASSIFIER_MODEL_PATH,
                                 classifier_config_path=CLASSIFIER_CONFIG_PATH,
                                 ingredients_config_path=INGREDIENTS_CONFIG_PATH,
                                 detector_model_path=DETECTOR_MODEL_PATH,
                                 detector_config_path=DETECTOR_CONFIG_PATH,
                                 detector_bbox_conf_threshold=DETECTOR_BBOX_CONF_THRESHOLD,
                                 blur_model_path=BLUR_MODEL_PATH,
                                 generator_model_path=GENERATOR_MODEL_PATH,
                                 generator_config_path=GENERATOR_CONFIG_PATH,
                                 debug=DEBUG)

INGREDIENTS_TEXT = image_processor.ingredients_text

# Allow using unverified SSL for image downloading
ssl._create_default_https_context = ssl._create_unverified_context


def generate_recipe(ingredients: list[str]) -> str:
    recipe = "Не могу найти напиток на изображении."
    if len(ingredients) > 0:
        if len(ingredients) > 1:
            recipe = ", ".join(ingredients[:-1])
            recipe = "".join(["Попробуем добавить ", recipe, " и ", ingredients[-1], "."])
        else:
            recipe = "".join(["Добавим только ", ingredients[-1], "."])
    return recipe


def generate_image(latent: Union[np.ndarray, None], ingr_list) -> str:
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)
    condition = np.zeros(len(INGREDIENTS_TEXT), dtype='float32')
    condition[ingr_list] = 1.
    image_processor.generate_to_file(latent, condition, full_filename)
    files_to_delete.append(full_filename)
    return full_filename


def predict(src, src_type: str) -> tuple[str, float, tuple[float, float, float, float], str]:
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)

    if src_type == "url":
        req = urllib.request.Request(src, data=None, headers=REQUEST_HEADERS)
        file = open(full_filename, "wb")
        try:
            file.write(urllib.request.urlopen(req).read(MAX_IMAGE_FILE_SIZE))
        except Exception as e_urllib:
            if DEBUG:
                print("urllib", e_urllib)
        finally:
            file.close()
    if src_type == "file":
        src.save(full_filename)

    ingredients, confidence, b_box = image_processor.predict(path=full_filename, threshold=CLASSIFIER_CONF_THRESHOLD)

    files_to_delete.append(full_filename)

    return generate_recipe(ingredients), confidence, b_box, filename


def blur_bounding_box(path: str, b_box: tuple[float, float, float, float]):
    image_processor.blur_bounding_box(path=path,
                                      b_box=b_box,
                                      power=VISUAL_BLUR_POWER,
                                      expansion=VISUAL_BLUR_BBOX_EXPANSION)

    if (DRAW_BBOX == "True") or (DRAW_BBOX == "Debug" and DEBUG):
        image_processor.draw_bounding_box(path=path,
                                          b_box=b_box,
                                          thickness=BBOX_LINE_THICKNESS,
                                          color=BBOX_LINE_COLOR)
