import json
from typing import Union, Tuple

import numpy as np
from openvino.runtime import Core
from PIL import Image, ImageDraw
from config import DEBUG
from utils import clip

import time


class Classifier:
    def __init__(self, onnx_model_path, model_config_path, ingredients_config_path: str):
        # Loading JSON config
        model_conf, \
        self.class_labels_ru, \
        self.ingredients_text = self.__import_json_model_config(model_config_path, ingredients_config_path)

        self.image_size = model_conf["IMAGE_SIZE"]
        self.crop_size = model_conf["CROP_SIZE"]

        # Loading the ONNX model
        open_vino_ = Core()
        model_onnx = open_vino_.read_model(model=onnx_model_path)
        self.model_onnx = open_vino_.compile_model(model=model_onnx, device_name="CPU")
        del model_onnx

    def __import_json_model_config(self, model_config_path: str, ingredients_config_path: str):
        # Opening model JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        # Opening ingredients JSON config
        with open(ingredients_config_path, 'r', encoding="utf-8") as file:
            ingedients_config = json.load(file)
        class_labels = ingedients_config["idx"]
        id2rus_genitive = ingedients_config["id2rus_genitive"]
        id2rus_nominative = ingedients_config["id2rus_nominative"]
        class_labels_rus = np.array([id2rus_genitive[idx] for idx in class_labels])
        ingredients_list = [id2rus_nominative[idx].capitalize() for idx in class_labels]
        return model_config, class_labels_rus, ingredients_list

    def predict_ingredients(self, path: str, threshold=0.5) -> tuple[list[int], float]:
        img = self.__open_resized_image(path, int(self.image_size), int(self.crop_size))

        if img is not None:
            logits = self.model_onnx([np.rollaxis(img, 2, 0)[None, :, :, :]])[self.model_onnx.output(0)][0]
            probs = 1 / (1 + np.exp(-logits))

            ingredients = self.class_labels_ru[(probs > threshold).nonzero()]

            pos_ind = (probs > threshold).nonzero()[0]
            neg_ind = (probs < threshold).nonzero()[0]
            confidence = np.prod(probs[pos_ind]) * np.prod(1 - probs[neg_ind])
            return ingredients, confidence
        return [], 0.

    def __open_resized_image(self, path: str, image_size: int, crop_size: int) -> Union[np.array, None]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None
        width, height = image.size  # Get dimensions
        size = round(min(width, height) / crop_size ** 0.5 * image_size ** 0.5)

        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

        return np.asarray(image.resize((image_size, image_size))) / 127.5 - 1.0


class Detector:
    def __init__(self, onnx_model_path, model_config_path):
        # Loading JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        self.image_size = model_config['IMAGE_SIZE']
        self.bbox_eccentricity_penalty_power = model_config['BBOX_ECCENTRICITY_PENALTY_POWER']
        self.bbox_size_penalty_power = model_config['BBOX_SIZE_PENALTY_POWER']
        self.bbox_expansion = 0.

        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.detector_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.detector_onnx.create_infer_request()
        del model_onnx

    def __open_image(self, path: str) -> np.array:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exception:
            if DEBUG:
                print(str(exception))
            return None
        return np.asarray(image.resize((self.image_size, self.image_size))) / 255

    def predict_bbox(self,
                     path: str,
                     threshold: float = 0.1,
                     ) -> Union[Tuple[float, float, float, float], None]:

        img = self.__open_image(path)

        if img is not None:

            inputs = [np.rollaxis(img, 2, 0)[None, :, :, :]]
            if DEBUG:
                print("Input shape", inputs[0].shape)
            outputs = self.infer_request.infer(inputs)[self.detector_onnx.output(0)][0]

            conf_filt = outputs[outputs[:, 4] > threshold * 0.5]
            confidence = conf_filt[:, 4] * conf_filt[:, 5]

            if confidence.max() > threshold:
                size_penalty = (conf_filt[:, 2] * conf_filt[:, 3]) ** self.bbox_size_penalty_power
                eccentricity_penalty = ((conf_filt[:, 0] - self.image_size / 2) ** 2 +
                                        (conf_filt[:, 1] - self.image_size / 2) ** 2) \
                                       ** -self.bbox_eccentricity_penalty_power
                penalties = size_penalty * eccentricity_penalty

                confidence_penalted = confidence * penalties

                bbox = conf_filt[confidence_penalted.argmax(), [0, 1, 2, 3]]
                # bbox = conf_filt[confidence.argmax(), [0, 1, 2, 3]]
                x_min = (bbox[0] - bbox[2] * 0.5 * self.bbox_expansion) / self.image_size
                y_min = (bbox[1] - bbox[3] * 0.5 * self.bbox_expansion) / self.image_size
                x_max = (bbox[0] + bbox[2] * 0.5 * self.bbox_expansion) / self.image_size
                y_max = (bbox[1] + bbox[3] * 0.5 * self.bbox_expansion) / self.image_size

                return x_min, y_min, x_max, y_max

        return None

    def add_bounding_box(self, path: str,
                         b_box: Union[Tuple[float, float, float, float], None],
                         thickness: float = 3.,
                         color: str = "#000000") -> None:

        x_min, y_min, x_max, y_max = b_box
        with Image.open(path).convert("RGB") as image:
            width, height = image.size
            x_min, x_max = round(clip(x_min, 0.01, 0.99) * width), round(clip(x_max, 0.01, 0.99) * width)
            y_min, y_max = round(clip(y_min, 0.01, 0.99) * height), round(clip(y_max, 0.01, 0.99) * height)

            line_width = round((width + height) * thickness * 2e-3)

            draw = ImageDraw.Draw(image)
            draw.line([(x_min, y_min),
                       (x_max, y_min),
                       (x_max, y_max),
                       (x_min, y_max),
                       (x_min, y_min)],
                      fill=color,
                      width=line_width,
                      joint='curve')
            image.save(path, "JPEG")


class BlurModel:
    def __init__(self, onnx_model_path):
        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.model_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.model_onnx.create_infer_request()
        self.model_output = self.model_onnx.output(0)
        del model_onnx

    def __generate_blur_mask(self, size: Tuple[int, int], b_box: Tuple[float, float, float, float]) -> np.array:
        height, width = size
        y_min, x_min, y_max, x_max = b_box

        x_min, x_max = round(x_min * width), round(x_max * width)
        y_min, y_max = round(y_min * height), round(y_max * height)

        result = np.zeros((width, height), dtype=np.float32)

        x = np.linspace(x_min / width, 0, x_min)
        y = np.linspace(y_min / height, 0, y_min)
        yv, xv = np.meshgrid(y, x)
        box1 = (1 / ((xv ** 2 + yv ** 2) ** 0.5 + 1)).astype(np.float32)

        box2 = 1 / (np.full((x_max - x_min, y_min), np.linspace(y_min / height, 0, y_min)) + 1)

        x = np.linspace(0, (width - x_max) / width, width - x_max)
        y = np.linspace(y_min / height, 0, y_min)
        yv, xv = np.meshgrid(y, x)
        box3 = (1 / ((xv ** 2 + yv ** 2) ** 0.5 + 1)).astype(np.float32)

        box4 = 1 / (np.full((y_max - y_min, width - x_max),
                            np.linspace(0, (width - x_max) / width, width - x_max)) + 1).T

        x = np.linspace(0, (width - x_max) / width, width - x_max)
        y = np.linspace(0, (height - y_max) / height, height - y_max)
        yv, xv = np.meshgrid(y, x)
        box5 = (1 / ((xv ** 2 + yv ** 2) ** 0.5 + 1)).astype(np.float32)

        box6 = 1 / (np.full((x_max - x_min, height - y_max),
                            np.linspace(0, (height - y_max) / height, height - y_max)) + 1)

        x = np.linspace(x_min / width, 0, x_min)
        y = np.linspace(0, (height - y_max) / height, height - y_max)
        yv, xv = np.meshgrid(y, x)
        box7 = (1 / ((xv ** 2 + yv ** 2) ** 0.5 + 1)).astype(np.float32)

        box8 = 1 / (np.full((y_max - y_min, x_min), np.linspace(x_min / width, 0, x_min)) + 1).T

        result[x_min: x_max, y_min: y_max] = 1.0

        result[0: x_min, 0: y_min] = box1
        result[x_min: x_max, 0: y_min] = box2
        result[x_max:, 0: y_min] = box3
        result[x_max:, y_min: y_max] = box4
        result[x_max:, y_max:] = box5
        result[x_min: x_max, y_max:] = box6
        result[0: x_min, y_max:] = box7
        result[0: x_min, y_min: y_max] = box8

        return result[None, None, :, :] ** 2

    def blur_bounding_box(self, path: str,
                          b_box: Union[Tuple[float, float, float, float], None]) -> None:
        with Image.open(path).convert("RGB") as image:
            mask = self.__generate_blur_mask(image.size, b_box)
            image = np.moveaxis(np.asarray(image, dtype=np.float32), 2, 0)[None, :, :]
            model_input = np.concatenate((image, mask), axis=1)
            init_time = time.time()
            blured_image = self.infer_request.infer([model_input])[self.model_output][0]
            print("Time=", time.time() - init_time)
            blured_image = np.moveaxis(blured_image, 0, 2).astype(np.uint8)
            image = Image.fromarray(blured_image)
            image.save(path, "JPEG")
