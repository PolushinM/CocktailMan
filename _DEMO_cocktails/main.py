import numpy as np
from PIL import Image
from openvino.runtime import Core
from math import log

image_size = 128
classes_path = 'model/classes.npy'
classes_ru_path = 'model/classes_ru.npy'
onnx_model_path = 'model/classifier.onnx'

class_labels = np.load(classes_path)
class_labels_ru = np.load(classes_ru_path)

ie = Core()
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")


def predict_ingredients(path: str, model: callable, classes: np.array, threshold=0.5) -> list:
    try:
        image = Image.open(path)
    except:
        return []
    width, height = image.size  # Get dimensions
    size = min(width, height)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    img = np.asarray(image.resize((image_size, image_size))) / 127.5 - 1.0
    logits = model([np.rollaxis(img, 2, 0)[None, 0:3, :, :]])[model.output(0)]
    result = (logits > -log(1/threshold - 0.999)).nonzero()[1]
    return classes[result]


def generate_recipe(ingredients: list) -> str:
    if len(ingredients) > 0:
        if len(ingredients) > 1:
            recipe = ', '.join(ingredients[:-1])
            recipe = ''.join(['Попробуем добавить ', recipe, ' и ', ingredients[-1],  '.'])
        else:
            recipe = ''.join(['Добавим только ', ingredients[-1], '.'])
        return recipe
    else:
        return'Не могу найти напиток на изображении.'


def getPrediction(img_path: str) -> str:
    ingredients = predict_ingredients(path=img_path, model=compiled_model_onnx, classes=class_labels_ru, threshold=0.3)
    return generate_recipe(ingredients)
