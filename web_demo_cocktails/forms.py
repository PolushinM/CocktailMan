from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileSize
from wtforms import SubmitField
from config import MAX_IMAGE_FILE_SIZE


class UploadForm(FlaskForm):
    input_file = FileField('', [FileSize(MAX_IMAGE_FILE_SIZE)])
    submit = SubmitField(label="Приготовить!")
