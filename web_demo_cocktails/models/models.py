import json
from typing import Union

import numpy as np
from openvino.runtime import Core
from PIL import Image, ImageDraw
from utils import clip


class ImageProcessor:
    def __init__(self,
                 max_moderated_size,
                 classifier_model_path,
                 classifier_config_path,
                 ingredients_config_path,
                 detector_model_path,
                 detector_config_path,
                 detector_bbox_conf_threshold,
                 blur_model_path,
                 generator_model_path,
                 generator_config_path,
                 debug):

        self.classifier = Classifier(onnx_model_path=classifier_model_path,
                                     model_config_path=classifier_config_path,
                                     ingredients_config_path=ingredients_config_path,
                                     debug=debug)

        self.detector = Detector(onnx_model_path=detector_model_path,
                                 model_config_path=detector_config_path,
                                 bbox_conf_threshold=detector_bbox_conf_threshold,
                                 debug=debug)

        self.blur_model = BlurModel(onnx_model_path=blur_model_path,
                                    debug=debug)

        self.generator = Generator(onnx_model_path=generator_model_path,
                                   model_config_path=generator_config_path,
                                   debug=False)

        self.max_size = max_moderated_size
        self.debug = debug
        self.blur_bbox_expansion = self.classifier.blur_bbox_expansion
        self.detector_bbox_expansion = self.classifier.detector_bbox_expansion
        self.blur_power = self.classifier.blur_power

    def predict(self, path: str, threshold: float) -> tuple[list[str], float, tuple[float, float, float, float]]:

        image = self.__open_image(path)
        if image is None:
            return [], 0., (0., 0., 0., 0.)

        b_box = self.detector.predict_bbox(image)
        if b_box is None:
            return [], 0., (0., 0., 0., 0.)

        crop_coordinates, crop_blur_bbox = self.__get_crop_coordinates(b_box, image.size)

        cropped_image = self.__crop_image(image=np.asarray(image), coordinates=crop_coordinates)

        classification_image = self.classifier.prepare_image(Image.fromarray(cropped_image.astype('uint8'), mode="RGB"))

        classification_image = self.blur_model.blur_image(image=classification_image,
                                                          blur_bbox=crop_blur_bbox,
                                                          power=self.blur_power)

        ingredients, confidence = self.classifier.classify_image(classification_image, threshold)

        if self.debug:
            classification_image = Image.fromarray(
                np.uint8(np.moveaxis(classification_image * 127.5 + 127.5, 0, 2)), mode="RGB")

            classification_image.save(
                "cache/classification_image.jpg",
                "JPEG", quality=100, subsampling=0)

        return ingredients, confidence, b_box

    def blur_bounding_box(self, path: str, b_box: tuple, power: float, expansion: float) -> None:
        if b_box is not None:
            with Image.open(path).convert("RGB") as image:

                width, height = image.size
                if width % 4 != 0:
                    image = image.crop((1, 0, width // 4 * 4 + 1, height))
                width, height = image.size
                if height % 4 != 0:
                    image = image.crop((0, 1, width, height // 4 * 4 + 1))

                image = np.moveaxis(np.asarray(image), 2, 0)
                blured_image = self.blur_model.blur_image(image,
                                                          blur_bbox=b_box,
                                                          power=power,
                                                          expansion=expansion)
                self.__save_image(blured_image, path)

    def generate_to_file(self, latent: Union[np.ndarray, None], condition: np.ndarray, path: str) -> None:
        image_array = self.generator.generate_image_array(latent, condition) * 255
        self.__save_image(image_array, path)

    def __open_image(self, path: str) -> Union[Image.Image, None]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exception:
            if self.debug:
                print(str(exception))
            return None
        # Size moderation
        width, height = image.size
        if (width > self.max_size) or (height > self.max_size):
            if width >= height:
                w = self.max_size
                h = round(self.max_size * height / width)
            else:
                h = self.max_size
                w = round(self.max_size * width / height)
            image = image.resize((w, h))
        return image

    def __save_image(self, image: Union[np.ndarray, Image.Image], path: str) -> None:
        if isinstance(image, np.ndarray):
            image = np.moveaxis(image, 0, 2).astype(np.uint8)
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image.save(path, "JPEG", quality=100, subsampling=0)
        else:
            raise ValueError("Argument 'image' must be numpy array or PIL Image")

    def draw_bounding_box(self, path: str,
                          b_box: Union[tuple[float, float, float, float], None],
                          thickness: float = 1.5,
                          color: str = "#000000") -> None:

        x_min, y_min, x_max, y_max = b_box

        image = self.__open_image(path)

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
        # image.save(path, "JPEG")
        self.__save_image(image, path)

    def __get_crop_coordinates(self, b_box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[
            tuple[int, int, int, int, int, int, int, int], tuple[float, float, float, float]]:

        width, height = image_size
        x_min_r, y_min_r, x_max_r, y_max_r = b_box
        x_center_r = x_min_r * 0.5 + x_max_r * 0.5
        y_center_r = y_min_r * 0.5 + y_max_r * 0.5
        bbox_width_r = x_max_r - x_min_r
        bbox_height_r = y_max_r - y_min_r

        result_size = round(
            max((x_max_r - x_min_r) * width, (y_max_r - y_min_r) * height) * self.detector_bbox_expansion / 2)

        result_x_min = round(x_center_r * width - result_size)
        result_pad_left = -result_x_min if result_x_min < 0 else 0
        result_x_min = result_x_min if result_x_min > 0 else 0

        result_y_min = round(y_center_r * height - result_size)
        result_pad_top = -result_y_min if result_y_min < 0 else 0
        result_y_min = result_y_min if result_y_min > 0 else 0

        result_x_max = round(x_center_r * width + result_size)
        result_pad_right = result_x_max - width if result_x_max > width else 0
        result_x_max = result_x_max if result_x_max < width else width

        result_y_max = round(y_center_r * height + result_size)
        result_pad_bottom = result_y_max - height if result_y_max > height else 0
        result_y_max = result_y_max if result_y_max < height else height

        crop_blur_bbox_y_min = ((y_center_r * height - bbox_height_r * height / 2 * self.blur_bbox_expansion) +
                                result_pad_top - result_y_min) / (result_size * 2)

        crop_blur_bbox_x_min = ((x_center_r * width - bbox_width_r * width / 2 * self.blur_bbox_expansion) +
                                result_pad_left - result_x_min) / (result_size * 2)

        crop_blur_bbox_y_max = ((y_center_r * height + bbox_height_r * height / 2 * self.blur_bbox_expansion) +
                                result_pad_top - result_y_min) / (result_size * 2)

        crop_blur_bbox_x_max = ((x_center_r * width + bbox_width_r * width / 2 * self.blur_bbox_expansion) +
                                result_pad_left - result_x_min) / (result_size * 2)

        crop_blur_bbox = crop_blur_bbox_x_min, crop_blur_bbox_y_min, crop_blur_bbox_x_max, crop_blur_bbox_y_max

        crop_coordinates = (result_x_min, result_pad_left,
                            result_y_min, result_pad_top,
                            result_x_max, result_pad_right,
                            result_y_max, result_pad_bottom)

        return crop_coordinates, crop_blur_bbox

    def __crop_image(self, image: np.ndarray, coordinates: tuple) -> np.ndarray:
        x_min, pad_left, y_min, pad_top, x_max, pad_right, y_max, pad_bottom = coordinates
        result = np.zeros((y_max - y_min + pad_top + pad_bottom, x_max - x_min + pad_left + pad_right, 3))
        crop = image[y_min:y_max, x_min:x_max, :]

        result[pad_top:y_max - y_min + pad_top, pad_left:x_max - x_min + pad_left, :] = crop

        if pad_left > 0:
            result[pad_top:y_max - y_min + pad_top, :pad_left + 1, :] = crop[:, 0:2, :].mean(axis=1)[:, None, :]

        if pad_right > 0:
            result[pad_top:y_max - y_min + pad_top, -pad_right - 1:, :] = crop[:, -3:-1, :].mean(axis=1)[:, None, :]

        if pad_top > 0:
            result[:pad_top + 1, :, :] = result[pad_top:pad_top + 2, :, :].mean(axis=0)[None, :, :]

        if pad_bottom > 0:
            result[-pad_bottom - 1:, :, :] = result[-pad_bottom - 3:-pad_bottom - 1, :, :].mean(axis=0)[None, :, :]

        return result

    @property
    def ingredients_text(self):
        return self.classifier.ingredients_text


class Classifier:
    def __init__(self, onnx_model_path, model_config_path, ingredients_config_path: str, debug: bool = False):
        self.debug = debug
        # Loading JSON config
        (model_conf,
         self.class_labels_ru,
         self.ingredients_text) = self.__import_json_model_config(model_config_path, ingredients_config_path)

        self.image_size = model_conf["IMAGE_SIZE"]
        self.blur_bbox_expansion = model_conf['BLUR_BBOX_EXPANSION']
        self.detector_bbox_expansion = model_conf['DETECTOR_BBOX_EXPANSION']
        self.blur_power = model_conf['BLUR_POWER']

        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.model_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        del model_onnx
        del open_vino_core

    def classify_image(self, image: np.ndarray, threshold: float):
        logits = self.model_onnx([image[None, :, :, :]])[self.model_onnx.output(0)][0]
        probs = 1 / (1 + np.exp(-logits))

        ingredients = self.class_labels_ru[(probs > threshold).nonzero()]

        pos_ind = (probs > threshold).nonzero()[0]
        neg_ind = (probs < threshold).nonzero()[0]
        confidence = np.prod(probs[pos_ind]) * np.prod(1 - probs[neg_ind])
        return ingredients, confidence

    def __import_json_model_config(self, model_config_path: str, ingredients_config_path: str) -> tuple[
                                                                                dict, np.ndarray, list[str]]:
        # Opening model JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        # Opening ingredients JSON config
        with open(ingredients_config_path, 'r', encoding="utf-8") as file:
            ingredients_config = json.load(file)
        class_labels = ingredients_config["idx"]
        id2rus_genitive = ingredients_config["id2rus_genitive"]
        id2rus_nominative = ingredients_config["id2rus_nominative"]
        class_labels_rus = np.array([id2rus_genitive[idx] for idx in class_labels])
        ingredients_list = [id2rus_nominative[idx].capitalize() for idx in class_labels]
        return model_config, class_labels_rus, ingredients_list

    def prepare_image(self, image: Image) -> np.ndarray:
        classification_image = image.resize((self.image_size, self.image_size))
        classification_image = np.moveaxis(np.asarray(classification_image), 2, 0) / 127.5 - 1.0
        return classification_image


class Detector:
    def __init__(self, onnx_model_path, model_config_path, bbox_conf_threshold, debug=False):
        # Loading JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        self.image_size = model_config['IMAGE_SIZE']
        self.bbox_eccentricity_penalty_power = model_config['BBOX_ECCENTRICITY_PENALTY_POWER']
        self.bbox_size_penalty_power = model_config['BBOX_SIZE_PENALTY_POWER']
        self.debug = debug
        self.bbox_conf_threshold = bbox_conf_threshold

        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.detector_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.detector_onnx.create_infer_request()
        del model_onnx
        del open_vino_core

    def predict_bbox(self, image: np.ndarray) -> Union[tuple[float, float, float, float], None]:

        if image is None:
            return None

        detection_image = np.moveaxis(np.asarray(image.resize((self.image_size, self.image_size))), 2, 0) / 255

        outputs = self.infer_request.infer([detection_image[None, :, :, :]])[self.detector_onnx.output(0)][0]

        conf_filt = outputs[outputs[:, 4] > self.bbox_conf_threshold * 0.3]
        confidence = conf_filt[:, 4] * conf_filt[:, 5]

        if (confidence.shape[0] > 0) and (confidence.max() > self.bbox_conf_threshold):
            size_penalty = (conf_filt[:, 2] * conf_filt[:, 3]) ** self.bbox_size_penalty_power
            eccentricity_penalty = ((conf_filt[:, 0] - self.image_size / 2) ** 2 +
                                    (conf_filt[:, 1] - self.image_size / 2) ** 2) \
                                   ** -self.bbox_eccentricity_penalty_power
            penalties = size_penalty * eccentricity_penalty

            confidence_penalted = confidence * penalties

            b_box = conf_filt[confidence_penalted.argmax(), [0, 1, 2, 3]]

            x_min = clip((b_box[0] - b_box[2] * 0.5) / self.image_size)
            y_min = clip((b_box[1] - b_box[3] * 0.5) / self.image_size)
            x_max = clip((b_box[0] + b_box[2] * 0.5) / self.image_size)
            y_max = clip((b_box[1] + b_box[3] * 0.5) / self.image_size)

            return x_min, y_min, x_max, y_max


class BlurModel:
    def __init__(self, onnx_model_path, debug=False):
        self.debug = debug
        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.model_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.model_onnx.create_infer_request()
        self.model_output = self.model_onnx.output(0)
        self.blur_request = self.model_onnx.create_infer_request()
        del model_onnx
        del open_vino_core

    def __generate_blur_mask(self, size: tuple[int, int],
                             b_box: tuple[float, float, float, float],
                             power: float,
                             expansion: float = 1.0) -> np.ndarray:
        height, width = size
        y_min_r, x_min_r, y_max_r, x_max_r = b_box
        box_width = x_max_r - x_min_r
        box_height = y_max_r - y_min_r

        x_min, x_max = round(clip(x_min_r - box_width * (expansion - 1) / 2) * width), round(
            clip(x_max_r + box_width * (expansion - 1) / 2) * width)
        y_min, y_max = round(clip(y_min_r - box_height * (expansion - 1) / 2) * height), round(
            clip(y_max_r + box_height * (expansion - 1) / 2) * height)

        result = np.ones((width, height), dtype=np.float32)

        x = np.linspace(x_min_r, 0, x_min)
        y = np.linspace(y_min_r, 0, y_min)
        yv, xv = np.meshgrid(y, x)
        box_top_left = ((xv ** 2 + yv ** 2) ** 0.5 + 1).astype(np.float32)

        box_left = np.full((x_max - x_min, y_min), np.linspace(y_min_r + 1, 1, y_min))

        x = np.linspace(0, 1 - x_max_r, width - x_max)
        y = np.linspace(y_min_r, 0, y_min)
        yv, xv = np.meshgrid(y, x)
        box_bottom_left = ((xv ** 2 + yv ** 2) ** 0.5 + 1).astype(np.float32)

        box_bottom = np.full((y_max - y_min, width - x_max), np.linspace(1, 2 - x_max_r, width - x_max)).T

        x = np.linspace(0, 1 - x_max_r, width - x_max)
        y = np.linspace(0, 1 - y_max_r, height - y_max)
        yv, xv = np.meshgrid(y, x)
        box_bottom_right = ((xv ** 2 + yv ** 2) ** 0.5 + 1).astype(np.float32)

        box_right = np.full((x_max - x_min, height - y_max), np.linspace(1, 2 - y_max_r, height - y_max))

        x = np.linspace(x_min_r, 0, x_min)
        y = np.linspace(0, 1 - y_max_r, height - y_max)
        yv, xv = np.meshgrid(y, x)
        box_top_right = ((xv ** 2 + yv ** 2) ** 0.5 + 1).astype(np.float32)

        box_top = np.full((y_max - y_min, x_min), np.linspace(x_min_r + 1, 1, x_min)).T

        result[: x_min, : y_min] = box_top_left
        result[x_min: x_max, : y_min] = box_left
        result[x_max:, : y_min] = box_bottom_left
        result[x_max:, y_min: y_max] = box_bottom
        result[x_max:, y_max:] = box_bottom_right
        result[x_min: x_max, y_max:] = box_right
        result[: x_min, y_max:] = box_top_right
        result[: x_min, y_min: y_max] = box_top

        result = (result ** -power).T[None, None, :, :]

        return result

    def blur_image(self, image: np.ndarray, blur_bbox: tuple, power: float, expansion: float = 1.0) -> np.ndarray:
        mask = self.__generate_blur_mask((image.shape[2], image.shape[1]), blur_bbox, power=power, expansion=expansion)
        mask = np.moveaxis(mask, 2, 3)
        blur_input = np.concatenate((image[None, ...], mask), axis=1)
        blured_image = self.blur_request.infer([blur_input])[self.model_output][0]
        return blured_image


class Generator:
    def __init__(self, onnx_model_path, model_config_path, debug=False):
        # Loading JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        self.image_size = model_config['IMAGE_SIZE']
        self.latent_size = model_config['LATENT_SIZE']
        self.contrast = model_config['CONTRAST']
        self.std = model_config['LATENT_SAMPLE_STD']
        self.debug = debug

        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.generator_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.generator_onnx.create_infer_request()
        self.model_output = self.generator_onnx.output(0)
        del model_onnx

    def generate_image_array(self, latent: Union[np.ndarray, None], condition: np.ndarray) -> np.ndarray:
        if latent is None:
            latent = np.random.normal(0, self.std, self.latent_size)
        inputs = np.concatenate((latent.astype('float32'), condition.astype('float32')), axis=0)[None, :]
        output = self.infer_request.infer([inputs])[self.model_output][0] * self.contrast - (self.contrast - 1) * 0.5
        return np.clip(output, a_max=1., a_min=0.)
