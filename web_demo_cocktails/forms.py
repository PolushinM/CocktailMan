"""Forms for web page."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileSize
from wtforms import StringField
from wtforms.widgets import html_params
from markupsafe import Markup

from config import MAX_IMAGE_FILE_SIZE


class ButtonWidget:
    """Renders a multi-line text area.
    `rows` and `cols` ought to be passed as keyword args when rendering.
    https://gist.github.com/doobeh/239b1e4586c7425e5114 """

    input_type = 'submit'

    html_params = staticmethod(html_params)

    def __call__(self, field, **kwargs):
        kwargs.setdefault('id', field.id)
        kwargs.setdefault('type', self.input_type)
        if 'value' not in kwargs:
            kwargs['value'] = field._value()

        params = self.html_params(name=field.name, **kwargs)
        label = kwargs.get('label', field.label.text)
        inner_template = kwargs.get('inner_template', field.label.text)
        template = f'<button {params}> {inner_template} {label}</button>'
        return Markup(template)


class ButtonField(StringField):
    """Custom button field for form."""
    widget = ButtonWidget()


class UploadForm(FlaskForm):
    """File upload form for main page."""
    input_file = FileField('', [FileSize(MAX_IMAGE_FILE_SIZE)])
    button = ButtonField()
