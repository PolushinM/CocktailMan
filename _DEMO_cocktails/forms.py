from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField


class UploadForm(FlaskForm):

    validators = [
        FileRequired(message='There was no file!'),
    ]

    input_file = FileField('', validators=validators)
    submit = SubmitField(label="Приготовить!")

