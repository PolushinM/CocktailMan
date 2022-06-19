
import os

import wget
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename

from main import getPrediction

from flask_bootstrap import Bootstrap
from forms import UploadForm

UPLOAD_FOLDER = 'static/'

app = Flask(__name__, static_folder="static")
bootstrap = Bootstrap(app)


app.secret_key = "546421349874624"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    recipe = ''
    file_exist = form.validate_on_submit()
    url_exist = "image_url" in request.form and len(request.form["image_url"]) > 0

    if request.method == 'POST':
        file = None
        if not file_exist and not url_exist:
            os.popen('cp static/placeholder.jpg static/download.jpg')
            flash('Файл отсутствует')
            return redirect('/')
        if file_exist:
            file = request.files['input_file']
            filename = secure_filename(file.filename)  # Use this werkzeug method to secure filename.
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg'))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg')
            try:
                recipe = getPrediction(full_filename)
            except:
                flash("Не могу прочитать изображение")
                return redirect('/')
            return render_template('index.html', form=form, recipe=recipe)
        if url_exist:
            image_url = request.form["image_url"]
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg')
            try:
                if os.path.exists(full_filename):
                    os.remove(full_filename)
                wget.download(image_url, full_filename)
                recipe = getPrediction(full_filename)
            except:
                os.popen('cp static/placeholder.jpg static/download.jpg')
                flash("Не могу прочитать изображение")
            return render_template('index.html', form=form, recipe=recipe)
    else:
        os.popen('cp static/placeholder.jpg static/download.jpg')
        return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
