import os

from flask import Flask, render_template, request, redirect, flash
from flask_bootstrap import Bootstrap
from forms import UploadForm

from main import get_prediction_file, ingredients_text, replace_image, calibrate_confidence, get_prediction_url, \
    uri_validator
from app_config import config

# Run flask app with Bootstrap extension
app = Flask(__name__, static_folder=config['static_folder'])
Bootstrap(app)
app.secret_key = config['secret_key']


@app.route('/', methods=['GET', 'POST'])
def index():
    recipe = ''
    confidence = 0.

    form = UploadForm()

    file_exist = form.validate_on_submit() and request.files['input_file'].filename and len(
        request.form["image_url"]) > 0
    url_exist = "image_url" in request.form and uri_validator(request.form["image_url"])

    if request.method == 'POST':
        if not file_exist and not url_exist:
            replace_image()
            flash('Файл отсутствует')
            return redirect('/')
        if file_exist:
            try:
                file = request.files['input_file']
                recipe, confidence = get_prediction_file(file)
            except Exception as e:
                flash("Не могу прочитать изображение")
                if config['debug']:
                    flash("file_exist_ " + str(e))
                return redirect('/')
            return render_index(form, recipe, confidence)
        if url_exist:
            try:
                image_url = request.form["image_url"]
                recipe, confidence = get_prediction_url(image_url)
            except Exception as e:
                replace_image()
                flash("Не могу прочитать изображение")
                if config['debug']:
                    flash("url_exist_ " + str(e))

            return render_index(form, recipe, confidence)
    else:
        replace_image()
        return render_index(form, recipe, confidence)


def render_index(form, recipe, confidence):
    confidence = calibrate_confidence(confidence)

    if confidence > 0.005:
        conf_text = f'Я уверен на {round(confidence * 100)}%!'
    else:
        conf_text = ''

    template = render_template('index.html',
                               form=form,
                               recipe=recipe,
                               ingr_text=ingredients_text,
                               conf_text=conf_text)
    return template
