import json
from math import log

import numpy as np
from PIL import Image
from openvino.runtime import Core

# Paths
config_path = 'config/'
onnx_model_path = 'model/classifier.onnx'

# Opening model JSON config
with open(''.join([config_path, 'model.json']), 'r') as f:
    model_conf = json.load(f)
image_size = int(model_conf['image_size'])
crop_size = int(model_conf['crop_size'])

# Opening ingredients JSON config
with open(''.join([config_path, 'ingredients.json']), 'r') as f:
    ingedients_config = json.load(f)
class_labels = ingedients_config["idx"]
id2rus_genitive = ingedients_config["id2rus_genitive"]
id2rus_nominative = ingedients_config["id2rus_nominative"]
class_labels_ru = np.array([id2rus_genitive[idx] for idx in class_labels])
ingredients_text = "\n".join([id2rus_nominative[idx].capitalize() for idx in class_labels])

# Loading the ONNX model
ie = Core()
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")


def predict_ingredients(path: str, model: callable, classes: np.array, threshold=0.5) -> list:
    try:
        image = Image.open(path)
    except:
        return []
    width, height = image.size  # Get dimensions
    size = round(min(width, height) / crop_size * image_size)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    img = np.asarray(image.resize((image_size, image_size))) / 127.5 - 1.0
    logits = model([np.rollaxis(img, 2, 0)[None, 0:3, :, :]])[model.output(0)]
    result = (logits > -log(1 / threshold - 0.999)).nonzero()[1]
    return classes[result]


def generate_recipe(ingredients: list) -> str:
    if len(ingredients) > 0:
        if len(ingredients) > 1:
            recipe = ', '.join(ingredients[:-1])
            recipe = ''.join(['Попробуем добавить ', recipe, ' и ', ingredients[-1], '.'])
        else:
            recipe = ''.join(['Добавим только ', ingredients[-1], '.'])
        return recipe
    else:
        return 'Не могу найти напиток на изображении.'


def getPrediction(img_path: str) -> str:
    ingredients = predict_ingredients(path=img_path, model=compiled_model_onnx, classes=class_labels_ru, threshold=0.25)
    return generate_recipe(ingredients)
