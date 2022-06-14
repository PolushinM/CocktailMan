from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from main import getPrediction
import os
import wget

UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__, static_folder="static")

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    file_exist = "file" in request.files
    url_exist = "image_url" in request.form

    if request.method == 'POST':
        if not file_exist and not url_exist:
            flash('No file part')
            return redirect(request.url)
        if file_exist:
            file = request.files['file']
        if url_exist:
            image_url = request.form["image_url"]
            filename = 'url_image.tmp'
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            wget.download(image_url, full_filename)
            label = getPrediction(full_filename)
            flash(label)
            flash(full_filename)
            return redirect('/')
        if file.filename == '' and "image_url" not in request.form:
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  # Use this werkzeug method to secure filename.
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # getPrediction(filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            label = getPrediction(full_filename)
            flash(label)
            flash(full_filename)
            return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
