import os
from flask import render_template, request, flash, send_from_directory

from forms import UploadForm

from main import predict, INGREDIENTS_TEXT, blur_bounding_box
from utils import get_confidence_text, uri_validator
from config import DEBUG, CACHE_FOLDER

import app

app = app.get_app()


@app.route('/', methods=['GET', 'POST'])
def index():
    recipe = ""
    confidence = 0.

    form = UploadForm()

    file_exist = form.validate_on_submit() and request.files['input_file'].filename
    url_exist = 'image_url' in request.form and uri_validator(request.form['image_url'])

    if request.method == "POST":
        if not file_exist and not url_exist:
            flash("Файл отсутствует")
            return render_index(form, recipe, confidence)
        if file_exist:
            try:
                file = request.files['input_file']
                recipe, confidence, bbox, filename = predict(file, src_type="file")
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(os.path.join(CACHE_FOLDER, filename), bbox)
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу прочитать изображение")
                if DEBUG:
                    print("file_exist_ " + str(exception))
                return render_index(form, recipe, confidence)
        if url_exist:
            try:
                image_url = request.form['image_url']
                recipe, confidence, bbox, filename = predict(image_url, src_type="url")
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(os.path.join(CACHE_FOLDER, filename), bbox)
                return render_index(form, recipe, confidence, filename)
            except Exception as exception:
                flash("Не могу прочитать изображение")
                if DEBUG:
                    print("url_exist_ " + str(exception))
                return render_index(form, recipe, confidence)

    return render_index(form, recipe, confidence)


@app.route('/cache/<path:filename>')
def download_file(filename):
    return send_from_directory(CACHE_FOLDER, filename, as_attachment=True)


def render_index(form, recipe, confidence, image_filename="placeholder"):
    template = render_template('index.html',
                               form=form,
                               recipe=recipe,
                               ingr_text=INGREDIENTS_TEXT,
                               conf_text=get_confidence_text(confidence),
                               image_filename=image_filename)
    return template
