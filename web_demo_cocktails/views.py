import os
import json
from flask import render_template, request, flash, send_from_directory

from forms import UploadForm

from main import predict, INGREDIENTS_TEXT, blur_bounding_box, generate_image
from utils import get_confidence_text, uri_validator
from config import DEBUG, CACHE_FOLDER

import app

app = app.get_app()

input_file = None
input_file_name = None
image_url = None
request_file_name = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global input_file, request_file_name
    global input_file_name
    global image_url

    form = UploadForm()

    if request.method == "GET":
        return render_template('index.html', form=form, ingr_list=INGREDIENTS_TEXT)

    if request.method == "POST":

        input_file = request.files.get('input_file')
        image_url = request.form.get('image_url')

        url_exist = (image_url is not None) and uri_validator(request.form['image_url'])
        file_exist = input_file is not None

        if DEBUG:
            print('file_exist: ', file_exist)
            print('input_file_name: ', input_file_name)
            print('input_file: ', input_file)
            print('url_exist', url_exist)
            print('image_url', image_url)

        if not file_exist and not url_exist:
            return send_request(message="Файл отсутствует")

        if file_exist:
            try:
                recipe, confidence, bbox, filename = predict(input_file, src_type="file")
                image_path = os.path.join(CACHE_FOLDER, filename)
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(image_path, bbox)
                return send_request(recipe=recipe, confidence=confidence, image_path=image_path)
            except Exception as exception:
                if DEBUG:
                    print("file_exist_ " + str(exception))
                return send_request(message="Не могу прочитать изображение")
        if url_exist:
            try:
                recipe, confidence, bbox, filename = predict(image_url, src_type="url")
                image_path = os.path.join(CACHE_FOLDER, filename)
                if bbox[2] * bbox[3] > 0.01:
                    blur_bounding_box(image_path, bbox)
                return send_request(recipe=recipe, confidence=confidence, image_path=image_path)
            except Exception as exception:
                if DEBUG:
                    print("url_exist_ " + str(exception))
                return send_request(message="Не могу прочитать изображение")
        return send_request()


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


def send_request(recipe="",
                 confidence=0.,
                 image_path=os.path.join("static/", "placeholder.jpg"),
                 message: str = ""
                 ) -> str:
    result = json.dumps({'image_path': image_path,
                         'flash_message': message,
                         'recipe': recipe,
                         'confidence': get_confidence_text(confidence)
                         })
    return result
