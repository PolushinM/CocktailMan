"""Page views site backend."""

import os
import json
import re

import numpy as np
from PIL import ImageColor
from flask import render_template, request, send_from_directory, flash
from forms import UploadForm
from flask_wtf import FlaskForm

from main import predict, INGREDIENTS_TEXT, LATENT_SIZE, generate_image, get_confidence_text
from utils import uri_validator
from config import CACHE_FOLDER
from logger import logger
import app

app = app.get_app()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Index page.
        Returns:
            Flask templates.render_template method.
    """
    form = UploadForm()

    if request.method == "GET":
        logger.debug(f"Index GET request received. INGREDIENTS_TEXT={INGREDIENTS_TEXT}")
        return render_index(form)

    if request.method == "POST":
        logger.debug("Index POST request received.")

        input_file = request.files.get('input_file')
        image_url = request.form.get('image_url')

        file_exist = form.validate_on_submit() and request.files['input_file'].filename and len(
            request.form['image_url']) > 0
        url_exist = 'image_url' in request.form and uri_validator(request.form['image_url'])

        logger.debug(f"input_file = {input_file}, "
                     f"image_url = {image_url}, "
                     f"file_exist = {file_exist}, "
                     f"url_exist = {url_exist}.")

        if not file_exist and not url_exist:
            logger.info("Not file_exist and not url_exist.")
            flash("Файл отсутствует.")
            return render_index(form)

        if file_exist:
            try:
                recipe, confidence, filename = predict(input_file, src_type="file")
                image_path = os.path.join(CACHE_FOLDER, filename)
                logger.debug(f"file_exist: recipe = {recipe}, "
                             f"confidence = {confidence}, "
                             f"filename = {filename}, "
                             f"image_path = {image_path}.")
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу прочитать изображение.")
                logger.error(f"file_exist exception: {str(exception)}")
                return render_index(form)
        if url_exist:
            try:
                recipe, confidence, filename = predict(image_url, src_type="url")
                image_path = os.path.join(CACHE_FOLDER, filename)
                logger.debug(f"url_exist: recipe = {recipe}, "
                             f"confidence = {confidence}, "
                             f"filename = {filename}, "
                             f"image_path = {image_path}.")
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу скачать изображение.")
                logger.error(f"url_exist exception: {str(exception)}")
                return render_index(form)
        return render_index(form)


@app.route('/generative_model', methods=['GET', 'POST'])
def generative_model():
    """Generative model page.
        Returns:
            Flask templates.render_template method.
    """
    if request.method == "GET":
        logger.debug(f"Generative_model GET request received. INGREDIENTS_TEXT={INGREDIENTS_TEXT}")
        return render_template('generative_model.html',
                               ingr_list=INGREDIENTS_TEXT,
                               image_filename='',
                               latent_size=LATENT_SIZE)

    if request.method == "POST":
        logger.debug("Generative_model POST request received.")
        form_list = [(item, request.form[item]) for item in request.form]
        ingr_list = list()
        ranges = [0] * LATENT_SIZE
        logger.debug(request.form)
        background = [0, 0, 0]

        for key, value in form_list:
            range_match = re.match(r'Range_(\d+)', key)
            ingr_match = re.match(r'Ingr_(\d+)', key)
            if ingr_match and value == "on":
                ingr_list.append(int(ingr_match.groups(0)[0]))
            if range_match:
                ranges[int(range_match.groups(0)[0]) - 1] = int(value) / 50 - 1
            if key == "Color":
                background = ImageColor.getcolor(value, "RGB")
        image_path = generate_image(latent=np.array(ranges), background=np.array(background), ingr_list=ingr_list)
        return json.dumps({'image_path': image_path})


@app.route('/cache/<path:filename>')
def download_file(filename):
    """Route for image in cache folder."""
    return send_from_directory(CACHE_FOLDER, filename, as_attachment=True)


def render_index(form: FlaskForm, recipe: str = "", confidence: float = 0., image_filename: str = "placeholder"):
    """Wrapper for Flask templates.render_template method.
        Args:
            form: FlaskForm for file input.
            recipe: text human-readable description of recipe.
            confidence: calibrated confidence of classifier.
            image_filename: image filename in cache folder.

        Returns:
            Flask templates.render_template method.
    """
    conf_text = get_confidence_text(confidence)
    template = render_template('index.html',
                               form=form,
                               recipe=recipe,
                               ingr_text=INGREDIENTS_TEXT,
                               conf_text=conf_text,
                               image_filename=image_filename)
    logger.debug(f"recipe = {recipe}, conf_text = {conf_text}")
    return template
