from flask import Flask

from config import STATIC_FOLDER, SECRET_KEY


def get_app():
    app = Flask(__name__, static_folder=STATIC_FOLDER)
    app.secret_key = SECRET_KEY
    return app
