import os
import ssl

import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image

from config import (CACHE_FOLDER, DEBUG, CLASSIFIER_CONF_THRESHOLD, MAX_FILE_SIZE, CLASSIFIER_CONFIG_PATH,
                    INGREDIENTS_CONFIG_PATH, CLASSIFIER_MODEL_PATH, REQUEST_HEADERS, DETECTOR_MODEL_PATH,
                    DETECTOR_CONFIG_PATH, BBOX_EXPANSION, BBOX_CONF_THRESHOLD, MAX_MODERATED_SIZE, BBOX_BLUR_POWER,
                    BLUR_MODEL_PATH)

from utils import get_random_filename, clear_cache
from models.models import Classifier, Detector, BlurModel

files_to_delete = []

classifier = Classifier(CLASSIFIER_MODEL_PATH, CLASSIFIER_CONFIG_PATH, INGREDIENTS_CONFIG_PATH)
INGREDIENTS_TEXT = classifier.ingredients_text

detector = Detector(DETECTOR_MODEL_PATH, DETECTOR_CONFIG_PATH)
detector.bbox_expansion = BBOX_EXPANSION

blur_model = BlurModel(BLUR_MODEL_PATH)
blur_model.blur_power = BBOX_BLUR_POWER

# Allow using unverified SSL for image downloading
ssl._create_default_https_context = ssl._create_unverified_context


def predict_ingredients(img_path: str) -> tuple[str, float]:
    ingredients, confidence = classifier.predict_ingredients(path=img_path, threshold=CLASSIFIER_CONF_THRESHOLD)
    return generate_recipe(ingredients), confidence


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


def predict_ingredients_from_url(image_url: str) -> tuple[str, float, str]:
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)
    req = urllib.request.Request(
        image_url,
        data=None,
        headers=REQUEST_HEADERS,
    )
    file = open(full_filename, "wb")
    try:
        file.write(urllib.request.urlopen(req).read(MAX_FILE_SIZE))
    except Exception as e_urllib:
        if DEBUG:
            print("urllib", e_urllib)
    finally:
        file.close()

    moderate_size(full_filename, MAX_MODERATED_SIZE)

    recipe, confidence = predict_ingredients(full_filename)
    files_to_delete.append(full_filename)
    return recipe, confidence, filename


def predict_ingredients_from_file(file):
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)
    file.save(full_filename)
    moderate_size(full_filename, MAX_MODERATED_SIZE)
    recipe, confidence = predict_ingredients(full_filename)
    files_to_delete.append(full_filename)
    print("files_to_delete: ", files_to_delete)
    return recipe, confidence, filename


def blur_bounding_box(path: str):
    b_box = detector.predict_bbox(path, threshold=BBOX_CONF_THRESHOLD)
    if b_box is not None:
        print("bbox=", b_box)
        blur_model.blur_bounding_box(path, b_box)
