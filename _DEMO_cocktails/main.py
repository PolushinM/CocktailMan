import numpy as np
from PIL import Image
from openvino.runtime import Core

image_size = 200
classes_path = 'model/classes.npy'
onnx_model_path = 'model/classifier.onnx'

class_labels = np.load(classes_path)

ie = Core()
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")


def predict_ingredients(path: str, model: callable, classes: np.array) -> list:
    image = Image.open(path)
    width, height = image.size  # Get dimensions
    size = min(width, height)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    img = np.asarray(image.resize((image_size, image_size))) / 255
    logits = model([np.rollaxis(img, 2, 0)[None, :, :, :]])[model.output(0)]
    result = (logits > 0).nonzero()[1]
    return classes[result]


def generate_recipe(ingredients: list) -> str:
    return ', '.join(ingredients)


def getPrediction(img_path: str) -> str:
    ingredients = predict_ingredients(path=img_path, model=compiled_model_onnx, classes=class_labels)
    return generate_recipe(ingredients)
