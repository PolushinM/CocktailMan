from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import SubmitField


class UploadForm(FlaskForm):
    input_file = FileField('')
    submit = SubmitField(label="Приготовить!")

