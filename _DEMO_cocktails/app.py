import os

import wget
from flask import Flask, render_template, request, redirect, flash

from main import getPrediction, ingredients_text

from flask_bootstrap import Bootstrap
from forms import UploadForm


app = Flask(__name__, static_folder="static")
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/images/'
app.secret_key = "546421349874624"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    recipe = ''
    form = UploadForm()
    file_exist = form.validate_on_submit()
    url_exist = "image_url" in request.form and len(request.form["image_url"]) > 0

    if request.method == 'POST':
        file = None
        if not file_exist and not url_exist:
            replace_image()
            flash('Файл отсутствует')
            return redirect('/')
        if file_exist:
            file = request.files['input_file']
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg'))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg')
            try:
                recipe = getPrediction(full_filename)
            except:
                flash("Не могу прочитать изображение")
                return redirect('/')
            return render_index(form, recipe)
        if url_exist:
            image_url = request.form["image_url"]
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg')
            try:
                if os.path.exists(full_filename):
                    os.remove(full_filename)
                wget.download(image_url, full_filename)
                recipe = getPrediction(full_filename)
            except:
                replace_image()
                flash("Не могу прочитать изображение")
            return render_index(form, recipe)
    else:
        replace_image()
        return render_index(form, recipe)


def replace_image():
    os.popen(f'cp {UPLOAD_FOLDER}placeholder.jpg {UPLOAD_FOLDER}download.jpg')


def render_index(form, recipe):
    return render_template('index.html', form=form, recipe=recipe, ingr_text=ingredients_text)


if __name__ == "__main__":
    app.run(debug=True)
