import json

import numpy as np
from openvino.runtime import Core
from PIL import Image


class Model(object):
    def __init__(self, onnx_model_path, model_config_path):
        # Loading JSON config
        self.model_conf, \
        self.class_labels_ru, \
        self.ingredients_text = self.__import_json_model_config(model_config_path)

        # Loading the ONNX model
        ie = Core()
        model_onnx = ie.read_model(model=onnx_model_path)
        self.model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
        return

    def __import_json_model_config(self, model_config_path: str):
        # Opening model JSON config
        with open(''.join([model_config_path, 'model.json']), 'r') as f:
            model_config = json.load(f)
        # Opening ingredients JSON config
        with open(''.join([model_config_path, 'ingredients.json']), 'r') as f:
            ingedients_config = json.load(f)
        class_labels = ingedients_config["idx"]
        id2rus_genitive = ingedients_config["id2rus_genitive"]
        id2rus_nominative = ingedients_config["id2rus_nominative"]
        class_labels_rus = np.array([id2rus_genitive[idx] for idx in class_labels])
        ingredients_txt = "\n".join([id2rus_nominative[idx].capitalize() for idx in class_labels])
        return model_config, class_labels_rus, ingredients_txt

    def predict_ingredients(self, path: str, threshold=0.5) -> tuple[list[int], float]:
        img = self.__open_resized_image(path, int(self.model_conf["image_size"]), int(self.model_conf["crop_size"]))

        logits = self.model_onnx([np.rollaxis(img, 2, 0)[None, :, :, :]])[self.model_onnx.output(0)][0]
        probs = 1 / (1 + np.exp(-logits))

        ingredients = self.class_labels_ru[(probs > threshold).nonzero()]

        pos_ind = (probs > threshold).nonzero()[0]
        neg_ind = (probs < threshold).nonzero()[0]
        confidence = np.prod(probs[pos_ind]) * np.prod(1 - probs[neg_ind])

        return ingredients, confidence

    def __open_resized_image(self, path: str, image_size: int, crop_size: int) -> np.array:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return [], 0.
        width, height = image.size  # Get dimensions
        size = round(min(width, height) / crop_size ** 0.5 * image_size ** 0.5)

        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

        return np.asarray(image.resize((image_size, image_size))) / 127.5 - 1.0
