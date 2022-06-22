import os
from math import pi, sin

import wget
import numpy as np
from PIL import Image
from openvino.runtime import Core

from app_config import config, import_json_model_config

# Loading JSON config
model_conf, class_labels_ru, ingredients_text = import_json_model_config(config["json_config_path"])

# Loading the ONNX model
ie = Core()
model_onnx = ie.read_model(model=config["onnx_model_path"])
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")


def get_prediction(img_path: str) -> tuple[str, float]:
    ingredients, confidence = predict_ingredients(path=img_path,
                                                  model=compiled_model_onnx,
                                                  classes=class_labels_ru,
                                                  image_size=int(model_conf["image_size"]),
                                                  crop_size=int(model_conf["crop_size"]),
                                                  threshold=config["threshold"])
    return generate_recipe(ingredients), confidence


def open_resized_image(path: str, image_size: int, crop_size: int) -> np.array:
    try:
        image = Image.open(path)
    except:
        return [], 0.
    width, height = image.size  # Get dimensions
    size = round(min(width, height) / crop_size * image_size)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    return np.asarray(image.resize((image_size, image_size))) / 127.5 - 1.0
    
    
def predict_ingredients(path: str, 
                        model: callable, 
                        classes: np.array, 
                        image_size: int, 
                        crop_size: int, 
                        threshold=0.5) -> tuple[list[int], float]:
    
    img = open_resized_image(path, image_size, crop_size)
    
    logits = model([np.rollaxis(img, 2, 0)[None, 0:3, :, :]])[model.output(0)][0]
    probs = 1 / (1 + np.exp(-logits))

    ingredients = classes[(probs > threshold).nonzero()]

    pos_ind = (probs > threshold).nonzero()[0]
    neg_ind = (probs < threshold).nonzero()[0]
    confidence = np.prod(probs[pos_ind]) * np.prod(1 - probs[neg_ind])

    return ingredients, confidence


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


def get_prediction_url(image_url: str) -> tuple[str, float]:
    full_filename = os.path.join(config["UPLOAD_FOLDER"], "download.jpg")
    if os.path.exists(full_filename):
        os.remove(full_filename)
    wget.download(image_url, full_filename)
    return get_prediction(full_filename)


def get_prediction_file(file):
    file.save(os.path.join(config["UPLOAD_FOLDER"], "download.jpg"))
    full_filename = os.path.join(config["UPLOAD_FOLDER"], "download.jpg")
    return get_prediction(full_filename)


def calibrate_confidence(confidence: float) -> float:
    return (confidence + sin(confidence * pi) / 5) ** 0.7


def replace_image():
    os.popen(f'cp {config["UPLOAD_FOLDER"]}placeholder.jpg {config["UPLOAD_FOLDER"]}download.jpg')
