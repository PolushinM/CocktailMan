"""
ML models, wrapped in own classes and image processor.

ImageProcessor: Manage whole image processing pipeline.
Classifier: Classify images, predict ingredients.
Detector: detect cocktail on image, crop image with padding.
BlurModel: generate blur mask and blur space around bounding box of cocktail.

Only ImageProcessor have to instantiate outside the module. Each model class manages reading ONNX model and
model json config from disk, model tuning-up and provides methods for model inference.
"""

import os
import json
from typing import Union

import numpy as np
from openvino.runtime import Core
from PIL import Image, ImageDraw
from utils import clip, log_error, log_debug


class ImageProcessor:
    """Manage whole image processing pipeline: reading, writing files, classification, detection,
    generation, blurring etc.

        Args:
            max_moderated_size: int, max size of images, which the model works with. If image width or
                height is greater than max_moderated_size, image will be resized after reading,
                before sending to ML model.
            jpeg_quality: int, [0, 100] PILlow JPEG quality for saving image.
            classifier_model_path: str, relative path to *.onnx classifier model file.
            classifier_config_path: str, relative path to *.json classifier model config.
            ingredients_config_path: str, relative path to *.json ingredients config
                (contents list of cocktail ingredients with names).
            detector_model_path: str, relative path to *.onnx detector model file.
            detector_config_path: str, relative path to *.json detector model config.
            detector_bbox_conf_threshold: float, min confidence for cocktail detection.
            blur_model_path: str, relative path to *.onnx blur model file.
            generator_model_path: str, relative path to *.onnx generator model file.
            generator_config_path: str, relative path to *.json generator model config.
            cache_folder: str, relative path to image cache folder.
            visual_blur_power: float, power of blur mask  gradient (0, +inf). Greater value - strong
                blurring right near bounding box, less value - lighter gradient.
            visual_blur_bbox_expansion: float, expansion (if >1, blurring begins farther from detected cocktail) or
                constriction (if <1, blurring begins inside bounding box).
            draw_bbox: str, whether to draw bounding box on picture. Values: True, False,
                Debug (bounding box will be drawn only if debug==true).
            bbox_line_thickness: float, thickness of bounding box drawn, typically 1..5.
            bbox_line_color: str, html color for bounding box drawing.
            debug: bool, debug mode."""

    def __init__(self,
                 max_moderated_size: int,
                 jpeg_quality: int,
                 classifier_model_path: str,
                 classifier_config_path: str,
                 ingredients_config_path: str,
                 detector_model_path: str,
                 detector_config_path: str,
                 detector_bbox_conf_threshold: float,
                 blur_model_path: str,
                 generator_model_path: str,
                 generator_config_path: str,
                 cache_folder: str,
                 visual_blur_power: float,
                 visual_blur_bbox_expansion: float,
                 draw_bbox: str,
                 bbox_line_thickness: float,
                 bbox_line_color: str,
                 debug: bool):
        """Init."""
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
                                   debug=debug)

        self.max_size = max_moderated_size
        self.jpeg_quality = jpeg_quality
        self.debug = debug
        self.blur_bbox_expansion = self.classifier.blur_bbox_expansion
        self.detector_bbox_expansion = self.classifier.detector_bbox_expansion
        self.blur_power = self.classifier.blur_power
        self.cache_folder = cache_folder
        self.visual_blur_power = visual_blur_power
        self.visual_blur_bbox_expansion = visual_blur_bbox_expansion
        self.draw_bbox = draw_bbox
        self.bbox_line_thickness = bbox_line_thickness
        self.bbox_line_color = bbox_line_color

        log_debug(f"Initialise Image Processor (IP): "
                  f"max_moderated_size = {max_moderated_size}, "
                  f"classifier_model_path = {classifier_model_path}, "
                  f"classifier_config_path = {classifier_config_path}, "
                  f"ingredients_config_path = {ingredients_config_path}, "
                  f"detector_model_path = {detector_model_path}, "
                  f"detector_config_path = {detector_config_path}, "
                  f"detector_bbox_conf_threshold = {detector_bbox_conf_threshold}, "
                  f"blur_model_path = {blur_model_path}, "
                  f"generator_model_path = {generator_model_path}, "
                  f"generator_config_path = {generator_config_path}, "
                  f"cache_folder = {cache_folder}, "
                  f"blur_bbox_expansion = {self.blur_bbox_expansion}, "
                  f"detector_bbox_expansion = {self.detector_bbox_expansion}, "
                  f"blur_power = {self.blur_power}, "
                  f"cache_folder = {cache_folder}, "
                  f"visual_blur_power = {self.visual_blur_power}, "
                  f"visual_blur_bbox_expansion = {self.visual_blur_bbox_expansion}, "
                  f"draw_bbox = {self.draw_bbox}, "
                  f"bbox_line_thickness = {self.bbox_line_thickness}, "
                  f"bbox_line_color = {self.bbox_line_color}"
                  )

    def predict_blur_save(self, path: str, threshold: float) -> tuple[list[str], float]:
        """Execute image processing pipeline: read from file, detect cocktail, crop image around
           cocktail, blur and resize crop for classification, classify (predict list f ingredients),
           blur space around bounding box on original image, save image.

            Args:
               path: str, path to image file.
               threshold: float, min model (multilabel classifier) confidence for including ingredient.
            Returns:
                Tuple: List of ingredients, confidence of classifying, relative coordinates of bounding box corners.
        """
        image = self.__open_image(path)
        log_debug(f"IP: Image opened at {path}")
        if image is None:
            return [], 0.

        ingredients, confidence, b_box = self.classify(image, threshold)
        log_debug(f"IP: Image classified: ingredients = {ingredients}, "
                  f"confidence = {confidence}, "
                  f"b_box = {b_box}")

        if b_box[2] * b_box[3] > 0.01:
            blurred_image = self.blur_model.blur_image(image=np.asarray(image),
                                                       blur_bbox=b_box,
                                                       power=self.visual_blur_power,
                                                       expansion=self.visual_blur_bbox_expansion)
            result_image = Image.fromarray(np.moveaxis(blurred_image, 0, 2).astype('uint8'), mode="RGB")

            if (self.draw_bbox == 'Debug' and self.debug) or (self.draw_bbox == "True"):
                log_debug(f"IP: draw bbox")
                result_image = self.draw_bounding_box(image=result_image,
                                                      b_box=b_box,
                                                      thickness=self.bbox_line_thickness,
                                                      color=self.bbox_line_color)
            log_debug(f"IP: blur bounding box. path = {path}, "
                      f"b_box = {b_box}, "
                      f"result_image.size={result_image.size}")
        else:
            result_image = image

        self.__save_image(result_image, path)
        log_debug(f"IP: Image saved at {path}")
        return ingredients, confidence

    def classify(self, image: Image.Image, threshold: float) \
            -> tuple[list[str], float, tuple[float, float, float, float]]:
        """Classify image (predict list of ingredients). Image classification pipeline: detect cocktail,
            crop image around predicted bounding box, blur image around bounding box, resize image for classifier,
            predict list of ingredients.

            Args:
               image: PIL Image object.
               threshold: float, min model (multilabel classifier) confidence for including ingredient.
            Returns:
                Tuple: List of ingredients, confidence of classifying, relative coordinates of bounding box corners.
        """
        b_box = self.detector.predict_bbox(image)
        if b_box is None:
            return [], 0., (0., 0., 0., 0.)
        log_debug(f"IP: classify b_box={b_box}")

        crop_coordinates, crop_blur_bbox = self.__get_crop_coordinates(b_box, image.size)
        log_debug(f"IP: crop_coordinates={crop_coordinates}, crop_blur_bbox={crop_blur_bbox}")

        cropped_image = self.__crop_image(image=np.asarray(image), coordinates=crop_coordinates)
        log_debug(f"IP: cropped_image.shape={cropped_image.shape}")

        classification_image = self.classifier.prepare_image(
            Image.fromarray(cropped_image.astype('uint8'), mode="RGB"))
        log_debug(f"IP: classification_image.shape={classification_image.size}")

        classification_image = self.blur_model.blur_image(image=classification_image,
                                                          blur_bbox=crop_blur_bbox,
                                                          power=self.blur_power)
        log_debug(f"IP: classification_image.size={classification_image.shape}")

        ingredients, confidence = self.classifier.classify_image(classification_image, threshold)

        if self.debug:
            path = os.path.join(self.cache_folder, "classification_image.jpg")
            try:
                Image.fromarray(np.uint8(
                    np.moveaxis(classification_image, 0, 2)), mode="RGB").save(path, "JPEG", quality=80)
                log_debug(f"Classification image saved at {path}")
            except Exception as exception:
                log_error(f"Unable to save classification image, exception: {str(exception)}")

        return ingredients, confidence, b_box

    def generate_to_file(self, latent: Union[np.ndarray, None], background: np.ndarray, condition: np.ndarray, path: str) -> None:
        """Generate image of cocktail and save to file.

           Args:
               latent: latent vector for generator, default: None
                   (None = generate from random multivariate normal distribution vector, E=1, D=1).
               background: background RGB color vector for generator.
               condition: recipe vector, contains ingredients for cocktail,
                   1 - ingredient present, 0 - ingredient missing. Length must match
                   number of ingredients in config.
               path: path for saving resulting image.
        """
        image_array = self.generator.generate_image(latent, background, condition) * 255
        self.__save_image(image_array, path)

    def __open_image(self, path: str) -> Union[Image.Image, None]:
        """Open image from file, resize, if necessary.

            Args:
                path: path to image file.
            Returns:
                PIL image object in case of successful opening, None in case of error.
        """
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exception:
            log_error(f"Unable open image, exception: {str(exception)}")
            return None
        # Size moderation
        width, height = image.size
        if (width > self.max_size) or (height > self.max_size) or \
                (width % 4 != 0) or (height % 4 != 0):
            if width >= height:
                w = round(self.max_size / 4) * 4
                h = round(self.max_size * height / width / 4) * 4
            else:
                h = round(self.max_size / 4) * 4
                w = round(self.max_size * width / height / 4) * 4
            image = image.resize((w, h))
            log_debug(f"IP: Open Image.size={image.size}")
        return image

    def __save_image(self, image: Union[np.ndarray, Image.Image], path: str) -> None:
        """Save image to file.

            Args:
                image: PIL image object or np.ndarray.
                path: path for saving image.
            """
        if isinstance(image, np.ndarray):
            image = np.moveaxis(image, 0, 2).astype(np.uint8)
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image.save(path, "JPEG", quality=self.jpeg_quality)
        else:
            raise ValueError("Argument 'image' must be numpy array or PIL Image object")

    def draw_bounding_box(self, image: Image.Image,
                          b_box: Union[tuple[float, float, float, float], None],
                          thickness: float = 1.5,
                          color: str = "#000000") -> Image.Image:
        """Draw bounding box on image.

            Args:
                image: PIL image object.
                b_box: relative coordinates of bounding box corners [0, 1].
                thickness: thickness of bounding box drawn, typically 1..5.
                color: html color for bounding box drawing, default #000000 (black).

            Returns:
                PIL image object with bounding box drawn.
            """
        x_min, y_min, x_max, y_max = b_box

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
        log_debug(f"IP: draw_bounding_box result.size={image.size}")
        return image

    def __get_crop_coordinates(self, b_box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[
        tuple[int, int, int, int, int, int, int, int], tuple[float, float, float, float]]:
        """Calculate coordinates for crop image around bounding box
           and new relative coordinates of bounding box corners

            Args:
                b_box: tuple of relative coordinates of bounding box corners on initial image.
                image_size: tuple (width, height) of image.

            Returns:
                1. Tuple of coordinates for cropping with paddings sizes:
                    (x_min, pad_left, y_min, pad_top, x_max, pad_right, y_max, pad_bottom).
                2. Tuple of bounding box corner coordinates: (x_min, y_min, x_max, y_max)
        """
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
        """Crop image.

            Args:
                image: image array of shape: (height, width, 3).
                coordinates: Tuple of coordinates for cropping with paddings sizes:
                    (x_min, pad_left, y_min, pad_top, x_max, pad_right, y_max, pad_bottom).

            Returns:
                image array of shape: (height, width, 3).
            """
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
        """List of text description of ingredients."""
        return self.classifier.ingredients_text


class Classifier:
    """Wrapper for multilabel classifier of ingredients ML model.

        Args:
            onnx_model_path: str, relative path to *.onnx classifier model file.
            model_config_path: str, relative path to *.json classifier model config.
            ingredients_config_path: str, relative path to *.json ingredients config
                (contents list of cocktail ingredients with names).
            debug: debug mode, default = False.
    """

    def __init__(self,
                 onnx_model_path: str,
                 model_config_path: str,
                 ingredients_config_path: str,
                 debug: bool = False):
        """Init."""
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
        """Predict list of ingredients with confidence.

            Args:
                image: image array of shape (3, width, height).
                threshold: min model (multilabel classifier) confidence for including ingredient.

            Returns:
                Tuple: list of ingredients and confidence value.
        """
        image = np.asarray(image) / 127.5 - 1.0
        logits = self.model_onnx([image[None, :, :, :]])[self.model_onnx.output(0)][0]
        probs = 1 / (1 + np.exp(-logits))

        ingredients = self.class_labels_ru[(probs > threshold).nonzero()]

        pos_ind = (probs > threshold).nonzero()[0]
        neg_ind = (probs < threshold).nonzero()[0]
        confidence = np.prod(probs[pos_ind]) * np.prod(1 - probs[neg_ind])
        return ingredients, confidence

    def __import_json_model_config(self, model_config_path: str, ingredients_config_path: str) -> \
            tuple[dict, np.ndarray, list[str]]:
        """Import model config from *.json files.

            Args:
                model_config_path: relative path to *.json classifier model config.
                ingredients_config_path: relative path to *.json ingredients config
                    (contents list of cocktail ingredients with names).
            Returns:
                Tuple of:
                1. model config dict.
                2. list of (str identifiers) ingredients labels.
                3. List of ingredients text names.
        """
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

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for classification.
            Args:
                image: PIL image object.

            Returns:
                Prepared for classification image (PIL image), resized to size of model input.
        """
        classification_image = image.resize((self.image_size, self.image_size))
        return classification_image


class Detector:
    """Wrapper for detector of cocktail ML model.

        Args:
            onnx_model_path: relative path to *.onnx detector model file.
            model_config_path: relative path to *.json detector model config.
            bbox_conf_threshold: min confidence for cocktail detection.
            debug: debug mode, default = False.
    """

    def __init__(self, onnx_model_path: str,
                 model_config_path: str,
                 bbox_conf_threshold: float,
                 debug: bool = False):
        """Init."""
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
        """Detect cocktail on image: predict bounding box corners coordinates.

            Args:
                image: image array of shape (width, height, 3).

            Returns:
                Tuple of relative coordinates: (x_min, y_min, x_max, y_max).
        """
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
    """Wrapper for blurring model

        Args:
            onnx_model_path: relative path to *.onnx blur model file.
            debug: debug mode, default = False.
    """

    def __init__(self, onnx_model_path: str, debug: bool = False):
        """Init."""
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
        """Generate blurring mask.

            Args:
                size: image size (height, width).
                b_box: bounding box corners relative coordinates.
                power: power of blur mask  gradient (0, +inf). Greater value - strong
                    blurring right near bounding box, less value - lighter gradient.
                expansion: expansion (if >1, blurring begins farther from detected cocktail) or
                    constriction (if <1, blurring begins inside bounding box).

            Returns:
                Array of shape (1, 1, width, height), where 1.0 value corresponds
                to the absence of blurring, 0.0 corresponds to maximum blurring.
        """
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

        result = np.moveaxis(result, 2, 3)

        return result

    def blur_image(self, image: np.ndarray, blur_bbox: tuple, power: float, expansion: float = 1.0) -> np.ndarray:
        """Make gradient blurring around bounding box.
            Args:
                image: image array of shape (width, height, 3).
                blur_bbox: tuple of relative coordinates of bounding box corners (x_min, y_min, x_max, y_max).
                power: power of blur mask  gradient (0, +inf). Greater value - strong
                    blurring right near bounding box, less value - lighter gradient.
                expansion: expansion (if >1, blurring begins farther from detected cocktail) or
                    constriction (if <1, blurring begins inside bounding box).

            Returns:
                Blurred image array of shape (3, width, height)
        """
        image = np.moveaxis(np.asarray(image), 2, 0)  # c*h*w
        mask = self.__generate_blur_mask((image.shape[2], image.shape[1]), blur_bbox, power=power, expansion=expansion)
        blur_input = np.concatenate((image[None, ...], mask), axis=1)
        blurred_image = self.blur_request.infer([blur_input])[self.model_output][0]
        return blurred_image


class Generator:
    """Wrapper for image generator ML model.
        Args:
            onnx_model_path: relative path to *.onnx generator model file.
            model_config_path: relative path to *.json generator model config.
            debug: debug mode, default = False.
    """

    def __init__(self, onnx_model_path: str, model_config_path: str, debug: bool = False):
        """Init."""
        # Loading JSON config
        with open(model_config_path, 'r', encoding="utf-8") as file:
            model_config = json.load(file)
        self.image_size = model_config['IMAGE_SIZE']
        self.latent_size = model_config['LATENT_SIZE']
        self.contrast = model_config['CONTRAST']
        self.soft_contrast = model_config['SOFTCONTRAST']
        self.std = model_config['LATENT_SAMPLE_STD']
        self.debug = debug
        # Loading the ONNX model
        open_vino_core = Core()
        model_onnx = open_vino_core.read_model(model=onnx_model_path)
        self.generator_onnx = open_vino_core.compile_model(model=model_onnx, device_name="CPU")
        self.infer_request = self.generator_onnx.create_infer_request()
        self.model_output = self.generator_onnx.output(0)
        del model_onnx
        del open_vino_core

    def generate_image(self, latent: Union[np.ndarray, None], background: np.ndarray, condition: np.ndarray) -> np.ndarray:
        """Generate image from latent vector and condition (recipe) vector.
            Args:
                latent: latent vector for generator, default: None
                    (None = generate from random multivariate normal distribution vector, E=1, D=1).
                background: background RGB color vector for generator.
                condition: recipe vector, contains ingredients for cocktail,
                    1 - ingredient present, 0 - ingredient missing. Length must match
                    number of ingredients in config.

            Returns:
                Image array of shape (3, width, height).
        """
        if latent is None:
            latent = np.random.normal(0, 1, self.latent_size)
        latent *= self.std
        background = np.concatenate((background / 127.5 - 1, background / 127.5 - 1)).astype('float32')

        inputs = np.concatenate((latent.astype('float32'),
                                 condition.astype('float32'),
                                 background), axis=0)[None, :]
        output = self.infer_request.infer([inputs])[self.model_output][0]
        return self.normalize(output)

    def normalize(self, in_image: np.ndarray):
        out_image = 1 / (1 + np.exp(-in_image * self.soft_contrast))
        out_image = out_image * self.contrast - (self.contrast - 1) * 0.5
        return np.clip(out_image, a_max=1., a_min=0.)
