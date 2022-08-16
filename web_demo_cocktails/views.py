import os
import json
from flask import render_template, request, send_from_directory, flash

from forms import UploadForm

from main import predict, INGREDIENTS_TEXT, blur_bounding_box, generate_image
from utils import get_confidence_text, uri_validator
from config import DEBUG, CACHE_FOLDER

import app

app = app.get_app()


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    if request.method == "GET":
        return render_index(form)

    if request.method == "POST":

        input_file = request.files.get('input_file')
        image_url = request.form.get('image_url')

        file_exist = form.validate_on_submit() and request.files['input_file'].filename and len(
            request.form['image_url']) > 0
        url_exist = 'image_url' in request.form and uri_validator(request.form['image_url'])

        if not file_exist and not url_exist:
            flash("Файл отсутствует")
            return render_index(form)

        if file_exist:
            try:
                recipe, confidence, bbox, filename = predict(input_file, src_type="file")
                image_path = os.path.join(CACHE_FOLDER, filename)
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(image_path, bbox)
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу прочитать изображение")
                if DEBUG:
                    print("file_exist_ " + str(exception))
                return render_index(form)
        if url_exist:
            try:
                recipe, confidence, bbox, filename = predict(image_url, src_type="url")
                image_path = os.path.join(CACHE_FOLDER, filename)
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(image_path, bbox)
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу скачать изображение")
                if DEBUG:
                    print("url_exist_ " + str(exception))
                return render_index(form)
        return render_index(form)


@app.route('/generative_model', methods=['GET', 'POST'])
def generative_model():
    if request.method == "GET":
        return render_template('generative_model.html',
                               ingr_list=INGREDIENTS_TEXT,
                               image_filename='')

    if request.method == "POST":
        ingr_list = [int(item) for item in request.form]
        image_path = generate_image(latent=None, ingr_list=ingr_list)
        return json.dumps({'image_path': image_path})


@app.route('/cache/<path:filename>')
def download_file(filename):
    return send_from_directory(CACHE_FOLDER, filename, as_attachment=True)


def render_index(form, recipe="", confidence=0., image_filename="placeholder"):
    template = render_template('index.html',
                               form=form,
                               recipe=recipe,
                               ingr_text=INGREDIENTS_TEXT,
                               conf_text=get_confidence_text(confidence),
                               image_filename=image_filename)
    return template
