import os
import ssl

import urllib.request
from werkzeug.utils import secure_filename

from config import CACHE_FOLDER, DEBUG, PROBA_THRESHOLD, MAX_FILE_SIZE, \
    MODEL_CONFIG_PATH, ONNX_MODEL_PATH, REQUEST_HEADERS

from utils import get_random_filename, clear_cache
from model import Model

files_to_delete = []

model = Model(ONNX_MODEL_PATH, MODEL_CONFIG_PATH)
ingredients_text = model.ingredients_text

# Allow using unverified SSL for image downloading
ssl._create_default_https_context = ssl._create_unverified_context


def get_prediction(img_path: str) -> tuple[str, float]:
    ingredients, confidence = model.predict_ingredients(path=img_path, threshold=PROBA_THRESHOLD)
    return generate_recipe(ingredients), confidence


def generate_recipe(ingredients: list) -> str:
    if len(ingredients) > 0:
        if len(ingredients) > 1:
            recipe = ", ".join(ingredients[:-1])
            recipe = "".join(["Попробуем добавить ", recipe, " и ", ingredients[-1], "."])
        else:
            recipe = "".join(["Добавим только ", ingredients[-1], "."])
        return recipe
    else:
        return "Не могу найти напиток на изображении."


def get_prediction_url(image_url: str) -> tuple[str, float, str]:
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
            print(e_urllib)
        pass
    finally:
        file.close()

    recipe, confidence = get_prediction(full_filename)
    files_to_delete.append(full_filename)
    return recipe, confidence, filename


def get_prediction_file(file):
    clear_cache(files_to_delete)
    filename = secure_filename(get_random_filename())
    full_filename = os.path.join(CACHE_FOLDER, filename)
    file.save(full_filename)
    recipe, confidence = get_prediction(full_filename)
    files_to_delete.append(full_filename)
    print(files_to_delete)
    return recipe, confidence, filename

